"""Joint audio/mapping training; every eligible window, deterministic resume."""
from dataclasses import asdict
from datetime import datetime, timezone
import math
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .config import ModelConfig, TrainConfig, CACHE
from .data import Dataset, UpdateSampler, collate
from .io import atomic_json, load_json, digest, telemetry
from .metrics import Metrics
from .model import Lyra, token_loss
from .runtime import device_for, amp, gpu_session, move, device_batches, random_state, restore_random, save_checkpoint, load_checkpoint, memory
from .tokenizer import Tokenizer, VERSION


def worker_init(_):
    torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def losses(model, b):
    hidden, beats = model(b)
    total, count = token_loss(model, hidden, b['labels'], indices=b.get('loss_indices'))
    raw = F.binary_cross_entropy_with_logits(beats.float(), b['beats'], reduction='none',
                                             pos_weight=beats.new_tensor([6., 12.], dtype=torch.float32))
    return total, count, (raw * b['beat_mask']).sum()


@torch.no_grad()
def validate(model, dataset, device, count=64):
    state = random_state()
    was_training = model.training
    model.eval()
    total, tokens, beat, weight = 0., 0, 0., 0.
    # Covers the complete held-out timeline, including middle/late sections.
    indices = np.random.default_rng(2026).choice(len(dataset), min(count, len(dataset)), replace=False)
    try:
        for index in indices:
            cpu = collate([dataset[int(index)]])
            w = float(cpu['beat_mask'].sum())
            with amp(device):
                a, n, b = losses(model, move(cpu, device))
            total += float(a); tokens += n; beat += float(b); weight += w
        return dict(token_loss=total / max(1, tokens), beat_loss=beat / max(1, weight),
                    windows=len(indices), target_tokens=tokens)
    finally:
        model.train(was_training)
        restore_random(state)


def train(dataset_path, run_dir, cfg=None, resume=None, progress=print, cancelled=lambda: False,
          limit_updates=0):
    cfg = cfg or TrainConfig()
    run = Path(run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    saved = load_checkpoint(resume) if resume else None
    if not saved and (run / 'last.pt').exists():
        raise ValueError('Ebben a mappában már van tanítás. Válaszd a Folytatás gombot.')
    if saved:
        original = TrainConfig(**saved['train_config'])
        for key in ('workers', 'ram_cache_gib', 'disk_cache_gb', 'min_free_gib', 'batch_size',
                    'device', 'memory_fraction', 'cpu_threads', 'cycles', 'steps', 'sample_every', 'sample_map_id',
                    'sample_audio_example'):
            setattr(original, key, getattr(cfg, key))
        cfg = original
        cfg.__post_init__()
    torch.set_num_threads(cfg.cpu_threads)
    device = device_for(cfg.device)
    manifest = load_json(dataset_path)
    if not manifest:
        raise ValueError('Előbb készítsd elő a teljes Lyra datasetet.')
    if saved and saved['dataset_hash'] != manifest['hash']:
        raise ValueError('A folytatáshoz az eredeti dataset szükséges.')
    if manifest.get('index_sha256') and digest(manifest['index']) != manifest['index_sha256']:
        raise ValueError('A rögzített Lyra-index megváltozott.')
    config = ModelConfig(**saved['model_config']) if saved else ModelConfig.tiny() if cfg.model == 'tiny' else ModelConfig()
    validation_ram = min(.5, cfg.ram_cache_gib * .25)
    options = dict(root=str(CACHE), gb=cfg.disk_cache_gb, ram_gib=(cfg.ram_cache_gib - validation_ram) / max(1, cfg.workers),
                   min_free_gib=cfg.min_free_gib, legacy='F:/osu_cache/runtime-arrays-v1')
    dataset = Dataset(manifest, 'train', config, options, cfg.seed)
    validation = Dataset(manifest, 'validation', config, {**options, 'ram_gib': validation_ram}, cfg.seed)
    if cfg.sample_every and cfg.sample_audio_example:
        from .examples import load_example
        load_example(cfg.sample_audio_example)
    elif cfg.sample_every and cfg.sample_map_id:
        validation.row(cfg.sample_map_id)
    target = cfg.steps or math.ceil(len(dataset) * cfg.cycles / cfg.effective_batch)
    if target < 1:
        raise ValueError('Üres tanítási cél.')
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    step = saved['step'] if saved else 0
    before = saved['elapsed_seconds'] if saved else 0.
    best = saved.get('best_validation') if saved else None
    best = best if best is not None else math.inf
    schedule_total = saved['schedule_total'] if saved else max(target, 100000 if cfg.model != 'tiny' else target)
    atomic_json(run / 'dataset.json', manifest)
    atomic_json(run / 'config.json', dict(model=asdict(config), training=asdict(cfg),
                                         target_steps=target, schedule_total=schedule_total))
    history = Metrics(run)
    history.truncate_after(step)
    started = time.monotonic()
    outcome, error = 'complete', None
    with gpu_session(device, cfg.memory_fraction):
        model = Lyra(len(Tokenizer()), config).to(device)
        if saved:
            model.load_state_dict(saved['model'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=.01,
                                     **({'fused': True} if device.type == 'cuda' else {'foreach': False}))
        def factor(n):
            if n < cfg.warmup:
                return (n + 1) / cfg.warmup
            f = min(1., (n - cfg.warmup) / max(1, schedule_total - cfg.warmup))
            return .1 + .9 * .5 * (1 + math.cos(math.pi * f))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        if saved:
            optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
            scaler.load_state_dict(saved['scaler']); restore_random(saved['random'])
        del saved
        params = sum(p.numel() for p in model.parameters())
        progress(f'Lyra: {params:,} együtt tanított paraméter; {len(dataset):,} train ablak; cél {target:,} update.')
        sampler = UpdateSampler(dataset, cfg.effective_batch, cfg.batch_size, step, cfg.seed, target)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=cfg.workers,
                            pin_memory=device.type == 'cuda', worker_init_fn=worker_init,
                            generator=torch.Generator().manual_seed(cfg.seed),
                            **(dict(prefetch_factor=2, persistent_workers=True) if cfg.workers else {}))
        iterator = iter(loader)
        last_save = time.monotonic()
        ema = history.latest().get('loss_ema')
        model.train()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        def state():
            return dict(format='lyra-1', tokenizer=VERSION, model_config=asdict(config), train_config=asdict(cfg),
                        model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        scaler=scaler.state_dict(), random=random_state(), schedule_total=schedule_total,
                        step=step, elapsed_seconds=before + time.monotonic() - started,
                        best_validation=best if math.isfinite(best) else None, dataset_hash=manifest['hash'],
                        generation_defaults=dict(AR=8., OD=7., CS=4., HP=5.), parameters=params,
                        coverage=dict(windows=len(dataset), visited=step * cfg.effective_batch,
                                      completed_cycles=step * cfg.effective_batch // len(dataset)))

        def sample():
            from .generation import training_sample
            destination = run / 'samples' / f'step-{step}'
            random_before = random_state()
            try:
                kind = 'külső hangrészlet példagenerálása' if cfg.sample_audio_example else 'teljes dalos példagenerálás'
                progress(f'Lyra {step:,}: {kind}.')
                output = training_sample(model, validation, destination, device, cancelled,
                        reference_map_id=cfg.sample_map_id, training_step=step, progress=progress,
                        audio_example=cfg.sample_audio_example)
                progress(f'Lyra {step:,}: példamap elkészült: {output}')
            except InterruptedError:
                raise
            except Exception as exc:
                # Preview failures must not discard optimizer updates or stop training.
                atomic_json(destination / 'rejected.json', dict(reason=str(exc), error_type=type(exc).__name__,
                            traceback=traceback.format_exc(), step=step, reference_map_id=cfg.sample_map_id,
                            audio_example=cfg.sample_audio_example))
                progress(f'Lyra {step:,}: a példagenerálás sikertelen: {exc}; a tanítás folytatódik.')
            finally:
                restore_random(random_before); model.train()

        if not (run / 'last.pt').exists():
            save_checkpoint(run / 'last.pt', state())
        try:
            # Recover a preview interrupted after its optimizer checkpoint was saved.
            destination = run / 'samples' / f'step-{step}'
            if (step and cfg.sample_every and step % cfg.sample_every == 0 and not cancelled()
                    and not any(destination.glob('*.osz')) and not (destination / 'rejected.json').exists()):
                sample()
            initial_step = step
            while step < target:
                if cancelled():
                    outcome = 'cancelled'; break
                if limit_updates and step - initial_step >= limit_updates:
                    outcome = 'checked'; break
                tick = time.monotonic()
                batches = []
                for _ in range(cfg.effective_batch // cfg.batch_size):
                    if cancelled():
                        break
                    batches.append(next(iterator))
                if cancelled():
                    outcome = 'cancelled'; break
                data_seconds = time.monotonic() - tick
                targets = sum(int((b['labels'] != -100).sum()) for b in batches)
                beat_weight = sum(float(b['beat_mask'].sum()) for b in batches)
                rng = random_state()
                retries = 0
                while True:
                    optimizer.zero_grad(set_to_none=True)
                    totals = torch.zeros(3, device=device)
                    for b in device_batches(batches, device):
                        if cancelled():
                            break
                        with amp(device):
                            ce, _, beat = losses(model, b)
                            loss = ce / max(1, targets) + .3 * beat / max(1., beat_weight)
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError('Nem véges loss; a korábbi lezárt mentés megmarad.')
                        scaler.scale(loss).backward()
                        totals += torch.stack((loss.detach(), ce.detach() / max(1, targets), beat.detach() / max(1., beat_weight)))
                    if cancelled():
                        outcome = 'cancelled'; break
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer); scaler.update()
                    if scaler.get_scale() >= old_scale:
                        break
                    retries += 1
                    if retries > 8:
                        raise FloatingPointError('Az FP16 skála nyolc próbából sem stabilizálódott.')
                    restore_random(rng)
                    progress('FP16 újrapróba ugyanazzal a batch-csel, csökkentett skálával.')
                if outcome == 'cancelled':
                    optimizer.zero_grad(set_to_none=True); break
                scheduler.step(); step += 1
                values = totals.detach().cpu().tolist()
                ema = values[0] if ema is None else .98 * ema + .02 * values[0]
                seconds = time.monotonic() - tick
                visited = step * cfg.effective_batch
                cache_workers = {}
                for batch in batches:
                    for meta, stats in zip(batch['meta'], batch['cache_stats']):
                        cache_workers[meta['worker']] = stats
                cache_totals = {key: sum(s.get(key, 0) for s in cache_workers.values())
                                for key in {k for s in cache_workers.values() for k in s}}
                value = dict(status='running', timestamp_utc=datetime.now(timezone.utc).isoformat(),
                             metrics_schema=1, step=step, target_steps=target, loss=values[0], loss_ema=ema,
                             token_loss=values[1], beat_loss=values[2], lr=scheduler.get_last_lr()[0],
                             elapsed_seconds=before + time.monotonic() - started, seconds_per_update=seconds,
                             target_tokens=targets, target_tokens_per_second=targets / max(seconds, .001),
                             data_seconds=data_seconds, training_seconds=max(0, seconds - data_seconds),
                             data_wait_fraction=data_seconds / max(seconds, .001),
                             examples_per_second=cfg.effective_batch / max(seconds, .001),
                             amp_retries=retries, gradient_norm=float(norm), grad_scale=scaler.get_scale(),
                             visited_windows=visited, completed_cycles=visited // len(dataset),
                             cycle_fraction=(visited % len(dataset)) / len(dataset), train_windows=len(dataset),
                             parameters=params, batch_size=cfg.batch_size, effective_batch=cfg.effective_batch,
                             cache=cache_totals, **memory(device))
                del b, ce, beat, loss, totals
                if step == 1 or step % cfg.validate_every == 0:
                    result = validate(model, validation, device, cfg.validation_windows)
                    value['validation'] = result
                    value['validation_token_loss'] = result['token_loss']
                    if result['token_loss'] < best:
                        best = result['token_loss']; save_checkpoint(run / 'best.pt', state())
                history.write(value)
                atomic_json(run / 'status.json', value)
                if step == 1 or step % 10 == 0:
                    progress(f'Lyra {step:,}/{target:,}: loss {values[0]:.4f}, {seconds:.2f} s/update, {visited / len(dataset):.2f} teljes kör.')
                if step % cfg.validate_every == 0 or time.monotonic() - last_save >= cfg.checkpoint_minutes * 60:
                    save_checkpoint(run / 'last.pt', state()); last_save = time.monotonic()
                if cfg.sample_every and step % cfg.sample_every == 0:
                    sample()
        except (KeyboardInterrupt, InterruptedError):
            outcome = 'cancelled'
        except Exception as exc:
            outcome, error = 'failed', str(exc)
            raise
        finally:
            optimizer.zero_grad(set_to_none=True)
            save_checkpoint(run / 'last.pt', state())
            # Explicitly shut down persistent workers on cancellation and errors.
            if hasattr(iterator, '_shutdown_workers'):
                iterator._shutdown_workers()
            atomic_json(run / 'status.json', dict(**{k: v for k, v in telemetry(run / 'status.json', {}).items()
                                                  if k not in ('status', 'step', 'target_steps', 'elapsed_seconds', 'best_validation',
                                                               'error', 'checkpoint', 'visited_windows', 'completed_cycles', 'parameters')},
                        status=outcome, step=step, target_steps=target,
                        elapsed_seconds=before + time.monotonic() - started,
                        best_validation=best if math.isfinite(best) else None, error=error,
                        checkpoint=str(run / 'last.pt'), visited_windows=step * cfg.effective_batch,
                        completed_cycles=step * cfg.effective_batch // len(dataset), parameters=params))
            progress(f'Lyra {outcome}: {step:,} lezárt update; mentés: {run / "last.pt"}')
    return str(run / 'last.pt')

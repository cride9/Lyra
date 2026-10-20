"""Real optimizer-step memory check at the maximum supported context."""
from dataclasses import asdict
from pathlib import Path
import time
import tempfile
import numpy as np
import torch
from . import audio
from .config import ModelConfig, HOME
from .io import atomic_json, load_json
from .model import Lyra
from .runtime import amp, device_for, gpu_session, memory
from .tokenizer import Tokenizer
from .training import losses


def benchmark(output=None, device='auto', tiny=False, updates=3, batch_size=1, progress=print, cancelled=lambda: False):
    device = device_for(device)
    config = ModelConfig.tiny() if tiny else ModelConfig()
    if device.type == 'cpu' and not tiny:
        raise ValueError('The full 120M-parameter memory benchmark requires CUDA; use the tiny diagnostic on CPU.')
    if updates < 1 or batch_size not in (1, 2, 4, 8):
        raise ValueError('At least one measured update is required.')
    torch.set_num_threads(2)
    path = Path(output or HOME / 'checks' / 'gpu.json')
    report = dict(model='Lyra', config=asdict(config), device=str(device), passed=False)
    with gpu_session(device):
        try:
            model = Lyra(len(Tokenizer()), config).to(device).train()
            report['parameters'] = sum(p.numel() for p in model.parameters())
            frames = 160 if tiny else int(np.ceil(config.audio_ms / audio.FRAME_MS / 4)) * 4
            length = 128 if tiny else config.max_tokens
            b = dict(mel=torch.randn(batch_size, 128, frames, device=device), phase=torch.zeros(batch_size, frames, 3, device=device),
                     beats=torch.zeros(batch_size, frames, 2, device=device), beat_mask=torch.ones(batch_size, frames, 2, device=device),
                     tokens=torch.randint(1, len(Tokenizer()), (batch_size, length), device=device),
                     labels=torch.randint(1, len(Tokenizer()), (batch_size, length), device=device),
                     summary=torch.zeros(batch_size, config.summary_tokens, 68, device=device),
                     summary_valid=torch.ones(batch_size, config.summary_tokens, dtype=torch.bool, device=device))
            b['labels'][:, :min(config.history_tokens, length // 2)] = -100
            count = int((b['labels'] != -100).sum())
            weight = float(b['beat_mask'].sum())
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                        **({'fused': True} if device.type == 'cuda' else {'foreach': False}))
            scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            durations, attempts, successes = [], 0, 0
            while successes < updates + 1:
                if cancelled():
                    raise InterruptedError('Memory benchmark stopped.')
                tick = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                with amp(device):
                    ce, _, beat = losses(model, b)
                    loss = ce / count + .3 * beat / weight
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                previous = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                attempts += 1
                if scaler.get_scale() >= previous:
                    successes += 1
                    if successes > 1:
                        durations.append(time.monotonic() - tick)
                    progress(f'Memory benchmark: {successes}/{updates + 1} complete optimizer updates.')
                if attempts > updates + 16:
                    raise ValueError('The memory benchmark FP16 update is unstable.')
            report.update(passed=True, measured_updates=len(durations), maximum_tokens=length,
                          audio_frames=frames, seconds_per_microbatch=float(np.mean(durations)),
                          target_tokens_per_second=count / float(np.mean(durations)), loss=float(loss.detach()),
                          batch_size=batch_size, examples_per_second=batch_size / float(np.mean(durations)),
                          gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None, **memory(device))
            report['headroom_ok'] = device.type != 'cuda' or report['free_vram_gib'] >= .75
            if not report['headroom_ok']:
                report['warning'] = 'The benchmark passed, but free VRAM fell below 0.75 GiB. Choose a smaller batch size.'
        except torch.cuda.OutOfMemoryError:
            report.update(error='The maximum context does not fit in the available dedicated VRAM.', **memory(device))
        finally:
            atomic_json(path, report)
    if not report['passed']:
        raise ValueError(report['error'])
    return report


def cache_benchmark(dataset_path, output=None):
    from .data import Dataset
    manifest = load_json(dataset_path)
    config = ModelConfig(**{**asdict(ModelConfig()), **manifest['model_context']})
    with tempfile.TemporaryDirectory(prefix='lyra-cache-check-') as tmp:
        ds = Dataset(manifest, 'validation', config, dict(root=tmp, gb=.1, ram_gib=.05, min_free_gib=0))
        tick = time.monotonic(); sample = ds[0]; cold = time.monotonic() - tick
        tick = time.monotonic()
        for _ in range(3):
            again = ds[0]
        warm = (time.monotonic() - tick) / 3
        torch.testing.assert_close(sample['mel'], again['mel'], rtol=0, atol=0)
        report = dict(cold_seconds=cold, warm_seconds=warm, speedup=cold / max(warm, 1e-6),
                      identical_mel=True, counters=dict(ds.cache.stats), scope='one real held-out window, data loading only')
    if output:
        atomic_json(output, report)
    return report

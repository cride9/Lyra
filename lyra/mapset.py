"""Six independently generated difficulties, one timing grid and one atomic .osz."""
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import re
import tempfile
import uuid
import zipfile
import numpy as np
import torch
from . import audio, mapio
from .config import ModelConfig
from .generation import GenerationConfig, candidate, difficulty, infer_timing, stage_audio
from .geometry import validate_map
from .io import atomic_json, digest
from .model import Lyra
from .quality import repetition_report
from .runtime import device_for, gpu_session, load_checkpoint
from .tokenizer import Tokenizer


@dataclass(frozen=True)
class Tier:
    name: str
    minimum: float
    maximum: float | None
    target: float
    ar: float
    od: float

    def contains(self, stars):
        return math.isfinite(stars) and stars >= self.minimum and (self.maximum is None or stars < self.maximum)


TIERS = (Tier('Easy', 0, 2, 1.5, 3, 3), Tier('Normal', 2, 2.7, 2.3, 5, 4),
         Tier('Hard', 2.7, 4, 3.3, 7, 6), Tier('Insane', 4, 5.3, 4.6, 8.5, 7.5),
         Tier('Expert', 5.3, 6.5, 5.9, 9.2, 8.5), Tier('Expert+', 6.5, None, 7.5, 9.7, 9))


def validate_targets(targets=None):
    targets = {} if targets is None else targets
    if not isinstance(targets, dict) or set(targets) - {t.name for t in TIERS}:
        raise ValueError('Mapset targets must use Easy, Normal, Hard, Insane, Expert, Expert+ keys.')
    result = {}
    for tier in TIERS:
        value = float(targets.get(tier.name, tier.target))
        if not tier.contains(value) or value > 20:
            raise ValueError(f'{tier.name}: target stars are outside this difficulty band.')
        result[tier.name] = value
    return result


def profile(tier, cfg):
    defaults = dict(AR=tier.ar, OD=tier.od, CS=4., HP=5.)
    return {key: defaults[key] if getattr(cfg, key.lower()) is None else getattr(cfg, key.lower())
            for key in defaults}


def version(entry):
    suffix = ' [OUT OF RANGE]' if not entry['in_band'] else ''
    return f'{entry["name"]} - {entry["measured_stars"]:.2f} stars{suffix}'


def export_mapset(maps, audio_path, destination, metadata, cancelled=lambda: False):
    if len(maps) != len(TIERS):
        raise ValueError('A complete mapset requires all six difficulties.')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    title = Path(audio_path).stem
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', title).strip('. ')[:80] or 'song'
    target = destination / f'{safe} - Lyra mapset - {uuid.uuid4().hex[:8]}.osz'
    temp = target.with_suffix('.tmp')
    try:
        with tempfile.TemporaryDirectory(prefix='.mapset-', dir=destination) as folder:
            folder = Path(folder)
            audio_name = stage_audio(audio_path, folder)
            for (bm, entry), tier in zip(maps, TIERS):
                if cancelled():
                    raise InterruptedError('Mapset export stopped.')
                bm = bm.copy()
                bm.general['AudioFilename'] = audio_name
                bm.metadata.update(Title=title, Artist='Unknown', Creator='Lyra', Version=version(entry),
                                   BeatmapID='0', BeatmapSetID='-1', Tags='lyra ai-generated mapset')
                path = folder / (tier.name.replace('+', 'Plus') + '.osu')
                mapio.write(bm, path)
                validate_map(mapio.read(path), metadata['duration_ms'])
            atomic_json(folder / 'generation.json', metadata)
            with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as archive:
                for path in folder.iterdir():
                    archive.write(path, path.name)
        if cancelled():
            raise InterruptedError('Mapset export stopped.')
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return str(target.resolve())


def build_difficulties(model, mel, duration, timing, cfg, targets, root, progress, cancelled,
                       on_update=lambda _: None):
    maps, entries = [], []
    for index, tier in enumerate(TIERS):
        if cancelled():
            raise InterruptedError('Mapset generation stopped.')
        settings, attempts, choices = profile(tier, cfg), [], []
        for attempt in range(cfg.candidates):
            if cancelled():
                raise InterruptedError('Mapset generation stopped.')
            seed = cfg.seed + index * 1000 + attempt
            subcfg = replace(cfg, stars=targets[tier.name], candidates=1, seed=seed)
            label = f'{tier.name} [{index+1}/6], candidate {attempt+1}/{cfg.candidates}'
            progress(label)
            try:
                bm = candidate(model, mel, duration, timing, subcfg, settings, mel_device(model), seed,
                               lambda message: progress(f'{label}: {message}'), cancelled)
                validate_map(bm, duration)
                stats, repeat = difficulty(bm), repetition_report(bm.objects)
                if not math.isfinite(stats['stars']):
                    raise ValueError('Non-finite measured difficulty.')
                in_band = tier.contains(stats['stars'])
                record = dict(name=tier.name, minimum=tier.minimum, maximum=tier.maximum,
                              requested_stars=subcfg.stars, measured_stars=stats['stars'], in_band=in_band,
                              warning=None if in_band else f'{tier.name}: outside requested star band',
                              seed=seed, difficulty_settings=settings, repetition=repeat)
                attempts.append(dict(seed=seed, valid=True, measured_stars=stats['stars'], in_band=in_band))
                choices.append(((not in_band, abs(stats['stars']-subcfg.stars), repeat['penalty'], seed), bm, record))
            except ValueError as exc:
                attempts.append(dict(seed=seed, valid=False, error=str(exc)))
            atomic_json(root / f'{tier.name.replace("+", "Plus")}-attempts.json', attempts)
        if not choices:
            raise ValueError(f'{tier.name}: no structurally valid candidate. See {root}.')
        _, bm, entry = min(choices, key=lambda choice: choice[0])
        entry['attempts'] = attempts
        bm.metadata['Version'] = version(entry)
        mapio.write(bm, root / f'{tier.name.replace("+", "Plus")}.osu')
        maps.append((bm, entry)); entries.append(entry)
        atomic_json(root / 'difficulties.json', entries)
        on_update(entries)
        progress(f'{tier.name}: {entry["measured_stars"]:.2f} stars' + (' - OUT OF RANGE' if not entry['in_band'] else ''))
    return maps, entries


def mel_device(model):
    return next(model.parameters()).device


def generate_mapset(checkpoint_path, inputs, output_dir, cfg=None, targets=None,
                    progress=print, cancelled=lambda: False):
    cfg = cfg or GenerationConfig()
    cfg.validate()
    targets = validate_targets(targets)
    if isinstance(inputs, (str, Path)):
        path = Path(inputs)
        inputs = sorted(p for p in path.rglob('*') if p.suffix.lower() in audio.EXTENSIONS) if path.is_dir() else [path]
    if not inputs:
        raise ValueError('No audio file selected.')
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', mode='mapset', outputs=[], files=[], failures=[])
    report_path = destination / 'mapset-report.json'
    atomic_json(report_path, report)
    torch.set_num_threads(2)
    try:
        saved = load_checkpoint(checkpoint_path)
        model_config = ModelConfig(**saved['model_config'])
        checkpoint_step, checkpoint_hash = saved['step'], digest(checkpoint_path)
        device = device_for(cfg.device)
        with gpu_session(device):
            model = Lyra(len(Tokenizer()), model_config).to(device)
            model.load_state_dict(saved['model']); model.eval(); del saved
            for source in inputs:
                root = destination / ('mapset-details-' + uuid.uuid4().hex[:8])
                root.mkdir()
                entry = dict(source=str(source), status='running', details=str(root))
                report['files'].append(entry)
                atomic_json(report_path, report)
                try:
                    if cancelled():
                        raise InterruptedError('Mapset generation stopped.')
                    progress(f'Reading {Path(source).name}')
                    pcm = audio.decode(source)
                    duration = len(pcm) * 1000 / audio.SR
                    mel = audio.spectrogram(pcm).astype(np.float16).astype(np.float32); del pcm
                    timing, timing_report = infer_timing(model, mel, duration, device, cfg, cancelled)
                    def update(entries):
                        entry['difficulties'] = entries
                        atomic_json(report_path, report)
                    maps, entries = build_difficulties(model, mel, duration, timing, cfg, targets, root,
                                                       progress, cancelled, update)
                    metadata = dict(model='Lyra', mode='mapset', checkpoint_step=checkpoint_step,
                                    checkpoint_sha256=checkpoint_hash, source=str(source), source_sha256=digest(source),
                                    duration_ms=duration, complete_song=True, timing=timing_report,
                                    targets=targets, settings=asdict(cfg), difficulties=entries)
                    output = export_mapset(maps, source, destination, metadata, cancelled)
                    atomic_json(root / 'generation.json', metadata)
                    entry.update(status='complete', output=output, difficulties=entries)
                    report['outputs'].append(output)
                except ValueError as exc:
                    entry.update(status='failed', error=str(exc))
                    report['failures'].append(dict(source=str(source), error=str(exc), details=str(root)))
                atomic_json(report_path, report)
        report['status'] = 'partial' if report['failures'] and report['outputs'] else 'complete' if report['outputs'] else 'failed'
        atomic_json(report_path, report)
        if not report['outputs']:
            raise ValueError('No complete mapset was generated. See '+str(report_path))
        return report['outputs']
    except BaseException as exc:
        if report['files'] and report['files'][-1]['status'] == 'running':
            report['files'][-1].update(status='cancelled' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed', error=str(exc))
        report['status'] = 'cancelled' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed'
        report['error'] = str(exc)
        atomic_json(report_path, report)
        raise

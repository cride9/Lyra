"""Frozen external audio excerpts for repeatable, map-free training previews."""
import math
from pathlib import Path
import subprocess
import uuid
import numpy as np
from . import audio
from .cache import ArrayCache
from .config import CACHE
from .io import atomic_json, digest, identity, load_json

FORMAT = 'lyra-audio-example-v1'


def load_example(path):
    spec = load_json(path)
    if not spec or spec.get('format') != FORMAT:
        raise ValueError('Érvénytelen külső hangpélda.')
    if digest(spec['audio_path']) != spec['audio_sha256']:
        raise ValueError('A rögzített példazene megváltozott.')
    return spec


def prepare_example(source, destination, start_ms, end_ms, *, generation=None, settings=None):
    from .generation import GenerationConfig
    from dataclasses import asdict
    if not all(math.isfinite(v) for v in (start_ms, end_ms)) or not 0 <= start_ms < end_ms:
        raise ValueError('A hangrészlet kezdete legyen korábbi a végénél.')
    source, destination = Path(source).resolve(), Path(destination).resolve()
    cfg = generation or GenerationConfig(stars=6.5, seed=2026, temperature=.6, candidates=1)
    cfg.validate()
    source_hash = digest(source)
    pcm = audio.decode(source)
    source_duration = len(pcm) * 1000 / audio.SR
    if end_ms > source_duration:
        raise ValueError('A kért részlet túlnyúlik a zenén.')
    source_pcm_hash = audio.pcm_hash(pcm)
    del pcm
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / 'example.json'
    if manifest.exists():
        spec = load_example(manifest)
        if (spec['source_sha256'], spec['source_range_ms'], spec['generation'], spec['difficulty_settings']) != (
                source_hash, [start_ms, end_ms], asdict(cfg), settings or dict(AR=9.4, OD=9., CS=4., HP=5.)):
            raise ValueError('Ez a példamappa már más beállítású hangrészletet tartalmaz.')
        return str(manifest)
    target = destination / 'audio.ogg'
    temp = destination / f'audio-{uuid.uuid4().hex}.ogg'
    try:
        command = [audio.ffmpeg(), '-nostdin', '-v', 'error', '-threads', '1', '-i', str(source),
                   '-ss', str(start_ms / 1000), '-t', str((end_ms - start_ms) / 1000),
                   '-map', '0:a:0', '-vn', '-c:a', 'libvorbis', '-q:a', '6', str(temp)]
        result = subprocess.run(command, capture_output=True, timeout=600,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            raise ValueError(result.stderr.decode(errors='replace'))
        duration = len(audio.decode(temp)) * 1000 / audio.SR
        if abs(duration - (end_ms - start_ms)) > 100:
            raise ValueError('A kivágott hang hossza nem egyezik a kért részlettel.')
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    def stamp(ms):
        seconds = round(ms / 1000)
        return f'{seconds // 60}m{seconds % 60:02d}'
    spec = dict(format=FORMAT, source_path=str(source), source_sha256=source_hash,
                source_pcm_hash=source_pcm_hash, source_duration_ms=source_duration,
                source_range_ms=[start_ms, end_ms], audio_path=str(target), audio_sha256=digest(target),
                duration_ms=duration, title=f'{source.stem} ({stamp(start_ms)}-{stamp(end_ms)})',
                generation=asdict(cfg), difficulty_settings=settings or dict(AR=9.4, OD=9., CS=4., HP=5.))
    atomic_json(manifest, spec)
    return str(manifest)


def generate_example(model, manifest, destination, device, cancelled=lambda: False, *,
                     training_step=None, progress=lambda _: None):
    from .generation import GenerationConfig, candidate, difficulty, export, infer_timing
    from .quality import repetition_report
    spec = load_example(manifest)
    cfg = GenerationConfig(**spec['generation'])
    cfg.validate()
    if cancelled():
        raise InterruptedError('Generálás leállítva.')
    cache = ArrayCache(root=str(CACHE), gb=180, ram_gib=.05, min_free_gib=8, legacy=None)
    key = identity([spec['audio_sha256'], audio.VERSION])
    value = cache.get('external-example-mel', key)
    if value is None:
        pcm = audio.decode(spec['audio_path'])
        value = dict(mel=audio.spectrogram(pcm).astype(np.float16))
        cache.put('external-example-mel', key, value)
    mel = value['mel'].astype(np.float32)
    was_training = model.training
    model.eval()
    try:
        progress('Timing felismerése a külső zenéből.')
        timing, timing_report = infer_timing(model, mel, spec['duration_ms'], device, cfg, cancelled)
        bm = candidate(model, mel, spec['duration_ms'], timing, cfg, spec['difficulty_settings'],
                       device, cfg.seed, progress, cancelled)
        metadata = dict(model='Lyra', training_step=training_step, example_source='external_audio',
                        title=spec['title'], source_path=spec['source_path'], source_sha256=spec['source_sha256'],
                        source_pcm_hash=spec['source_pcm_hash'], source_range_ms=spec['source_range_ms'],
                        audio_sha256=spec['audio_sha256'], duration_ms=spec['duration_ms'],
                        complete_song=False, complete_excerpt=True, reference_timing=False,
                        reference_map_id=None, timing=timing_report, seed=cfg.seed,
                        requested_stars=cfg.stars, measured_stars=difficulty(bm)['stars'],
                        repetition=repetition_report(bm.objects), settings=spec['generation'],
                        difficulty_settings=spec['difficulty_settings'])
        return export(bm, spec['audio_path'], destination, metadata)
    finally:
        model.train(was_training)

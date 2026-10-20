"""Full-song Lyra generation from one self-contained checkpoint."""
from dataclasses import asdict, dataclass
import copy
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid
import zipfile
import numpy as np
import torch
from . import audio, mapio
from .config import ModelConfig, CACHE
from .context import conditioning
from .geometry import validate_map, validate_object
from .io import atomic_json, digest, source_parts, read_bytes
from .model import Lyra
from .quality import generation_bias, repetition_report
from .runtime import amp, device_for, gpu_session, load_checkpoint
from .timing import estimate
from .tokenizer import Tokenizer, Grammar


@dataclass
class GenerationConfig:
    stars: float = 5.
    preset: str = 'Auto'
    aim: int | None = None
    streams: int | None = None
    rhythm: int | None = None
    seed: int = 42
    candidates: int = 3
    temperature: float = .8
    top_p: float = .95
    bpm: float | None = None
    offset: float | None = None
    timing_map: str | None = None
    ar: float | None = None
    od: float | None = None
    cs: float | None = None
    hp: float | None = None
    device: str = 'auto'

    def styles(self):
        presets = {'Auto': {}, 'Aim': dict(aim=2, streams=0), 'Streams': dict(aim=0, streams=2),
                   'Rhythm': dict(rhythm=2), 'Custom': dict(aim=self.aim, streams=self.streams, rhythm=self.rhythm)}
        if self.preset not in presets:
            raise ValueError('Ismeretlen stílus.')
        values = dict(aim=None, streams=None, rhythm=None, **{})
        values.update(presets[self.preset])
        if any(v is not None and v not in (0, 1, 2) for v in values.values()):
            raise ValueError('A stílusérték 0, 1, 2 vagy Auto lehet.')
        return values

    def validate(self):
        if not 0 <= self.stars <= 20 or not 1 <= self.candidates <= 16 or not 0 < self.top_p <= 1 or self.temperature < 0:
            raise ValueError('Érvénytelen generálási beállítás.')
        for key in ('ar', 'od', 'cs', 'hp'):
            v = getattr(self, key)
            if v is not None and not 0 <= v <= 10:
                raise ValueError('A difficulty beállítás 0–10 között lehet.')
        self.styles()


def difficulty(bm):
    import rosu_pp_py as rosu
    parsed = rosu.Beatmap(content=mapio.dumps(bm))
    if parsed.is_suspicious():
        raise ValueError('A difficulty-mérő gyanús mapot jelzett.')
    attrs = rosu.Difficulty(lazer=False).calculate(parsed)
    return dict(stars=float(attrs.stars), aim=float(attrs.aim), speed=float(attrs.speed))


def sample_token(logits, allowed, cfg, generator, bias=None):
    if not allowed:
        raise ValueError('Nincs érvényes folytatás.')
    indices = torch.tensor(allowed, device=logits.device)
    values = logits[indices].float()
    if bias is not None:
        values = values + bias
    if not torch.isfinite(values).all():
        raise ValueError('Nem véges generálási logit.')
    if cfg.temperature == 0:
        return allowed[int(values.argmax())]
    probabilities, order = (values / cfg.temperature).softmax(0).sort(descending=True)
    probabilities[probabilities.cumsum(0) - probabilities >= cfg.top_p] = 0
    index = order[torch.multinomial(probabilities, 1, generator=generator)]
    return int(indices[index].item())


@torch.inference_mode()
def section(model, mel, bm, start, end, duration, cfg, settings, device, generator,
            cancelled=lambda: False):
    tok = Tokenizer()
    ctx = conditioning(bm, start, end, duration, cfg.stars, cfg.styles(), settings, model.config)
    base, prefix = ctx['base'], ctx['prefix']
    busy = max([start, *[o.time + bm.duration(o) for o in bm.objects if o.time < start],
                *[b for a, b in bm.breaks if a < start]])
    grammar = Grammar(tok, base, start, end, duration, busy)
    features = audio.window(mel, base, model.config.audio_ms)
    origin = round(base / audio.FRAME_MS) * audio.FRAME_MS
    phase = audio.phase_features(bm, origin, features.shape[1])
    with amp(device):
        memory, valid, _ = model.encode(torch.from_numpy(features)[None].to(device),
                torch.from_numpy(phase)[None].to(device), torch.from_numpy(ctx['summary'])[None].to(device),
                torch.from_numpy(ctx['summary_valid'])[None].to(device))
        logits, caches = model.step(torch.tensor([prefix], device=device), memory, valid)
    offset, event, retries = len(prefix), [], 0
    old_grammar, old_logits, old_caches, old_offset = copy.copy(grammar), logits, caches, offset
    exhausted = False
    while offset < model.config.max_tokens:
        if cancelled():
            raise InterruptedError('Generálás leállítva.')
        remaining = model.config.max_tokens - offset
        exhausted = grammar.state == 'event' and remaining < 32 and grammar.minimum < end
        allowed = grammar.allowed(remaining)
        indices = torch.tensor(allowed, device=device)
        bias = generation_bias(bm.objects, grammar.state, indices, tok) if grammar.kind == 'CIRCLE' else None
        token = sample_token(logits[0, -1], allowed, cfg, generator, bias)
        grammar.consume(token); event.append(token)
        if token == tok.ids['EOS']:
            break
        if token == tok.ids['END']:
            scratch = mapio.Beatmap(timing=copy.deepcopy(bm.timing), difficulty=dict(bm.difficulty))
            try:
                tok.decode(event, scratch, base)
                scratch.validate(duration)
                if scratch.objects:
                    validate_object(scratch.objects[0], settings['CS'])
            except (ValueError, IndexError, OverflowError):
                retries += 1
                if retries > 3:
                    raise ValueError('Három újrapróba után is hibás objektumgeometria.')
                grammar, logits, caches, offset = copy.copy(old_grammar), old_logits, old_caches, old_offset
                event = []
                continue
            bm.objects.extend(scratch.objects); bm.breaks.extend(scratch.breaks); bm.timing = scratch.timing
            event, retries = [], 0
        with amp(device):
            logits, caches = model.step(torch.tensor([[token]], device=device), memory, valid, caches, offset)
        offset += 1
        if token == tok.ids['END']:
            old_grammar, old_logits, old_caches, old_offset = copy.copy(grammar), logits, caches, offset
    if grammar.state not in ('done', 'event'):
        raise ValueError('A tokenkeret félbevágott egy objektumot.')
    if exhausted or offset >= model.config.max_tokens:
        events = [o.time for o in bm.objects if o.time >= start] + [a for a, _ in bm.breaks if a >= start]
        if not events:
            raise ValueError('Nem haladt előre a generálás.')
        return max(events) + 1
    return end


def candidate(model, mel, duration, timing, cfg, settings, device, seed,
              progress=print, cancelled=lambda: False, span=None):
    bm = mapio.Beatmap(timing=copy.deepcopy(timing))
    for key, value in settings.items():
        bm.difficulty[dict(AR='ApproachRate', OD='OverallDifficulty', CS='CircleSize', HP='HPDrainRate')[key]] = str(value)
    bm.metadata.update(Creator='Lyra', Tags='lyra ai-generated')
    start, finish = span or (0, math.ceil(duration))
    generator = torch.Generator(device=device).manual_seed(seed)
    cursor, iterations = start, 0
    while cursor < min(finish, duration):
        if cancelled():
            raise InterruptedError('Generálás leállítva.')
        end = min(math.ceil(duration), finish, cursor + model.config.output_ms)
        next_cursor = section(model, mel, bm, cursor, end, duration, cfg, settings, device, generator, cancelled)
        if next_cursor <= cursor:
            raise ValueError('A generálás nem halad előre.')
        cursor = next_cursor; iterations += 1
        progress(f'Generálás: {min(cursor, duration) / max(1, duration):.0%} · {len(bm.objects):,} objektum')
        if iterations > max(100, duration / 20):
            raise ValueError('Túl sok sikertelen szakaszfolytatás.')
    if not bm.objects:
        raise ValueError('A modell még üres mapot generált.')
    validate_map(bm, duration)
    return bm


@torch.inference_mode()
def infer_timing(model, mel, duration, device, cfg, cancelled=lambda: False):
    if cfg.timing_map:
        return [p for p in mapio.read(cfg.timing_map).timing if p.uninherited], dict(reference=cfg.timing_map)
    if cfg.bpm:
        return estimate(np.zeros((1, 2)), bpm=cfg.bpm, offset=cfg.offset)
    scores = np.zeros((mel.shape[1], 2), np.float32)
    for start in range(0, math.ceil(duration), model.config.output_ms):
        if cancelled():
            raise InterruptedError('Timing felismerés leállítva.')
        base = max(0, start - model.config.history_ms)
        x = torch.from_numpy(audio.window(mel, base, model.config.audio_ms))[None].to(device)
        with amp(device):
            _, _, beats = model.encode(x)
        values = beats[0].float().sigmoid().cpu().numpy()
        left, right = round(start / audio.FRAME_MS), min(len(scores), round((start + model.config.output_ms) / audio.FRAME_MS))
        origin = round(base / audio.FRAME_MS)
        scores[left:right] = values[left - origin:right - origin]
    return estimate(scores, offset=cfg.offset)


def export(bm, audio_path, destination, metadata):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    song_title = metadata.get('title') or metadata.get('reference_title') or Path(audio_path).stem
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', song_title).strip('. ')[:80] or 'song'
    step_label = f' step {metadata["training_step"]}' if metadata.get('training_step') is not None else ''
    version = f'Lyra{step_label} {metadata["measured_stars"]:.2f} stars'
    name = f'{title} - {version} - {uuid.uuid4().hex[:6]}'
    target = destination / (name + '.osz')
    temp = target.with_suffix('.tmp')
    try:
        with tempfile.TemporaryDirectory(prefix='export-', dir=CACHE) as folder:
            folder = Path(folder)
            suffix = Path(audio_path).suffix.lower()
            audio_name = 'audio' + suffix if suffix in ('.mp3', '.ogg') else 'audio.ogg'
            if suffix in ('.mp3', '.ogg'):
                shutil.copyfile(audio_path, folder / audio_name)
            else:
                proc = subprocess.run([audio.ffmpeg(), '-nostdin', '-v', 'error', '-threads', '1', '-i', str(audio_path),
                        '-map', '0:a:0', '-c:a', 'libvorbis', '-q:a', '6', str(folder / audio_name)], capture_output=True,
                        timeout=600, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if proc.returncode:
                    raise ValueError(proc.stderr.decode(errors='replace'))
            bm.general['AudioFilename'] = audio_name
            bm.metadata.update(Title=song_title, Creator='Lyra', Version=version,
                               Tags='lyra ai-generated', BeatmapID='0', BeatmapSetID='-1')
            mapio.write(bm, folder / 'Lyra.osu')
            mapio.read(folder / 'Lyra.osu').validate()
            atomic_json(folder / 'generation.json', metadata)
            with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as z:
                for file in folder.iterdir():
                    z.write(file, file.name)
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return str(target.resolve())


def generate(checkpoint_path, inputs, output_dir, cfg=None, progress=print, cancelled=lambda: False):
    cfg = cfg or GenerationConfig()
    cfg.validate()
    torch.set_num_threads(2)
    saved = load_checkpoint(checkpoint_path)
    config = ModelConfig(**saved['model_config'])
    device = device_for(cfg.device)
    checkpoint_hash = digest(checkpoint_path)
    defaults = saved.get('generation_defaults', dict(AR=8., OD=7., CS=4., HP=5.))
    settings = {k: getattr(cfg, k.lower()) if getattr(cfg, k.lower()) is not None else v for k, v in defaults.items()}
    if isinstance(inputs, (str, Path)):
        source = Path(inputs)
        inputs = sorted(p for p in source.rglob('*') if p.suffix.lower() in audio.EXTENSIONS) if source.is_dir() else [source]
    if not inputs:
        raise ValueError('Nincs kiválasztott zenefájl.')
    outputs, failures = [], []
    with gpu_session(device):
        model = Lyra(len(Tokenizer()), config).to(device)
        model.load_state_dict(saved['model']); model.eval()
        checkpoint_step = saved['step']
        del saved
        for audio_path in inputs:
            if cancelled():
                raise InterruptedError('Generálás leállítva.')
            try:
                progress(f'Zene beolvasása: {Path(audio_path).name}')
                pcm = audio.decode(audio_path)
                duration = len(pcm) * 1000 / audio.SR
                mel = audio.spectrogram(pcm).astype(np.float16).astype(np.float32); del pcm
                timing, timing_report = infer_timing(model, mel, duration, device, cfg, cancelled)
                candidates, errors = [], []
                for attempt in range(cfg.candidates):
                    try:
                        bm = candidate(model, mel, duration, timing, cfg, settings, device, cfg.seed + attempt, progress, cancelled)
                        stats = difficulty(bm)
                        repeat = repetition_report(bm.objects)
                        candidates.append((abs(stats['stars'] - cfg.stars) + repeat['penalty'] * .5,
                                           bm, stats, cfg.seed + attempt, repeat))
                    except ValueError as exc:
                        errors.append(str(exc))
                if not candidates:
                    raise ValueError('; '.join(errors))
                _, bm, stats, seed, repeat = min(candidates, key=lambda v: v[0])
                metadata = dict(model='Lyra', checkpoint_sha256=checkpoint_hash, checkpoint_step=checkpoint_step,
                                requested_stars=cfg.stars, measured_stars=stats['stars'], seed=seed,
                                timing=timing_report, repetition=repeat, complete_song=True, settings=asdict(cfg),
                                difficulty_miss=abs(stats['stars'] - cfg.stars) > .5, candidate_errors=errors)
                outputs.append(export(bm, audio_path, output_dir, metadata))
                progress(f'Elkészült: {stats["stars"]:.2f}★, {len(bm.objects):,} objektum.')
            except ValueError as exc:
                failures.append(dict(audio=str(audio_path), error=str(exc)))
    atomic_json(Path(output_dir) / 'generation-report.json', dict(outputs=outputs, failures=failures))
    if not outputs:
        raise ValueError('Nem készült érvényes map: ' + '; '.join(f['error'] for f in failures))
    return outputs


@torch.inference_mode()
def training_sample(model, dataset, destination, device, cancelled=lambda: False, *,
                    reference_map_id=None, training_step=None, progress=lambda _: None, audio_example=None):
    if audio_example:
        from .examples import generate_example
        return generate_example(model, audio_example, destination, device, cancelled,
                                training_step=training_step, progress=progress)
    from .io import readonly
    if not reference_map_id:
        wid = int(dataset.entries[0, 0])
        with readonly(dataset.manifest['index']) as db:
            reference_map_id = db.execute('SELECT map_id FROM windows WHERE id=?', (wid,)).fetchone()[0]
    # A fixed training example can belong to any split; validation membership stays unchanged.
    row = dataset.row(reference_map_id)
    reference = dataset.beatmap(row)
    count = math.ceil(row['duration_ms'] / audio.FRAME_MS)
    # Preview uses the same immutable mel tiles as training; no separate model.
    normalized, _, _ = dataset.mel_window(row, 0, count)
    mel = normalized * 7 - 5
    cfg = GenerationConfig(stars=row['stars'], temperature=.6, candidates=1, seed=2026)
    was_training = model.training
    model.eval()
    try:
        bm = candidate(model, mel, row['duration_ms'], [p for p in reference.timing if p.uninherited],
                cfg, row['settings'], device, cfg.seed, progress, cancelled)
        bm.metadata.update(Title=row.get('title', 'Lyra'), Artist=row.get('artist', ''))
        source, member = source_parts(row['audio_path'])
        metadata = dict(measured_stars=difficulty(bm)['stars'], requested_stars=row['stars'],
                        reference_timing=True, complete_song=True, model='Lyra', training_step=training_step,
                        reference_map_id=row['id'], reference_beatmap_id=reference.metadata.get('BeatmapID'),
                        reference_map_path=row['map_path'], reference_split=row['split'],
                        reference_title=row.get('title'), reference_artist=row.get('artist'),
                        reference_version=row.get('version'), duration_ms=row['duration_ms'],
                        seed=cfg.seed, settings=asdict(cfg), difficulty_settings=row['settings'])
        if member is None:
            return export(bm, source, destination, metadata)
        CACHE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=CACHE) as folder:
            source = Path(folder) / ('audio' + Path(member).suffix)
            source.write_bytes(read_bytes(row['audio_path']))
            return export(bm, source, destination, metadata)
    finally:
        model.train(was_training)

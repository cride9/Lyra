"""Import every usable map from the frozen corpus; index ALL Lyra windows."""
from dataclasses import asdict
from functools import lru_cache
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import itertools
import multiprocessing
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import subprocess
import numpy as np
from . import audio, mapio
from .config import ModelConfig, CACHE
from .context import windows, conditioning
from .io import atomic_json, load_json, identity, readonly, signature, read_bytes, parse_map, source_parts, digest
from .tokenizer import Tokenizer, VERSION


def _prepare_batch(batch, model_config):
    config = ModelConfig(**model_config)
    result = []
    for number, row in batch:
        try:
            if row.get('excluded'):
                raise ValueError('A rögzített dataset kézzel kizárt mapja.')
            raw = read_bytes(row['map_path'], 8 * 1024 ** 2)
            if hashlib.sha256(raw).hexdigest() != row['map_hash']:
                raise ValueError('A map megváltozott az eredeti előkészítés óta.')
            bm = parse_map(row['map_path'])
            spans = windows(bm, row['duration_ms'], config)
            keep = {k: row[k] for k in ('id', 'map_path', 'map_hash', 'audio_path', 'audio_hash',
                    'duration_ms', 'stars', 'settings', 'styles', 'split', 'group')}
            for k in ('bytes_sha256', 'timing_labels', 'title', 'artist', 'version'):
                keep[k] = row.get(k)
            keep['map_signature'], keep['audio_signature'] = signature(row['map_path']), signature(row['audio_path'])
            result.append((number, row['id'], keep, spans, None))
        except (ValueError, OSError, KeyError) as exc:
            result.append((number, row['id'], None, None, str(exc)))
    return result


def _process_records(rows, config, workers, cancelled):
    if not workers:
        for batch in iter(lambda: list(itertools.islice(rows, 32)), []):
            yield from _prepare_batch(batch, asdict(config))
        return
    # Bounded submissions: never pickle the entire 148k-map corpus into RAM.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = deque()
        def submit():
            batch = list(itertools.islice(rows, 32))
            if batch:
                pending.append(pool.submit(_prepare_batch, batch, asdict(config)))
            return bool(batch)
        for _ in range(workers * 2):
            if not submit():
                break
        while pending:
            if cancelled():
                for future in pending:
                    future.cancel()
                raise InterruptedError('Előkészítés elmentve; folytatható.')
            yield from pending.popleft().result()
            submit()


def map_dict(bm):
    return asdict(bm)


def map_from_dict(v):
    v = dict(v)
    v['timing'] = [mapio.TimingPoint(**p) for p in v['timing']]
    v['objects'] = [mapio.HitObject(**o) for o in v['objects']]
    return mapio.Beatmap(**v)


def prepare(source, destination, config=None, progress=print, cancelled=lambda: False, workers=4):
    from filelock import FileLock
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / 'prepare.lock'), timeout=0):
        return _prepare(source, destination, config, progress, cancelled, workers)


def _prepare(source, destination, config, progress, cancelled, workers):
    config = config or ModelConfig()
    source = Path(source)
    if source.is_dir():
        source = source / 'manifest.json'
    manifest = load_json(source)
    if not manifest or 'index' not in manifest:
        raise ValueError('A már előkészített teljes dataset manifest.json fájlja szükséges.')
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    geometry = {k: asdict(config)[k] for k in ('audio_ms', 'history_ms', 'output_ms', 'history_tokens', 'max_tokens')}
    labels = load_json(source.parent / 'training-dataset.json', {}).get('labels', {})
    fingerprint = identity([manifest['hash'], manifest.get('index_hash'), identity(labels), geometry, VERSION])
    published = load_json(root / 'manifest.json')
    if published:
        if published['hash'] != fingerprint:
            raise ValueError('Ez a Lyra-adatkönyvtár más datasethez vagy kontextushoz tartozik.')
        return published
    path = root / ('index.sqlite' if (root / 'index.sqlite').exists() else 'index.build.sqlite')
    db = sqlite3.connect(path)
    try:
        db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)')
        saved = db.execute("SELECT value FROM metadata WHERE key='hash'").fetchone()
        if saved and saved[0] != fingerprint:
            raise ValueError('Az előkészítés forrása megváltozott. Válassz új célkönyvtárat.')
        db.execute("INSERT OR IGNORE INTO metadata VALUES('hash',?)", (fingerprint,))
        db.execute('CREATE TABLE IF NOT EXISTS maps(id TEXT PRIMARY KEY,split TEXT,group_id TEXT,payload TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS windows(id INTEGER PRIMARY KEY,map_id TEXT,start INTEGER,end INTEGER,tokens INTEGER,split TEXT)')
        db.execute('CREATE INDEX IF NOT EXISTS map_windows ON windows(map_id)')
        db.execute('CREATE TABLE IF NOT EXISTS rejected(id TEXT PRIMARY KEY,error TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS audio_labels(audio_hash TEXT PRIMARY KEY,payload TEXT)')
        db.commit()
        existing = {r[0] for r in db.execute('SELECT id FROM maps UNION SELECT id FROM rejected')}
        audio_splits, group_splits = {}, {}
        with readonly(manifest['index']) as src:
            total = src.execute('SELECT COUNT(*) FROM records').fetchone()[0]
            def rows():
                for number, (payload,) in enumerate(src.execute('SELECT payload FROM records ORDER BY id'), 1):
                    row = json.loads(payload)
                    if row['id'] in labels:
                        row.update(labels[row['id']])
                    for seen, key in ((audio_splits, row['audio_hash']), (group_splits, row['group'])):
                        if seen.setdefault(key, row['split']) != row['split']:
                            raise ValueError('Train/validáció/test dalátfedés a forrásadatokban.')
                    if row['id'] not in existing:
                        yield number, row
            completed = 0
            for number, mid, keep, spans, error in _process_records(iter(rows()), config, workers, cancelled):
                if keep:
                    db.execute('INSERT INTO maps VALUES(?,?,?,?)', (mid, keep['split'], keep['group'], json.dumps(keep, ensure_ascii=False)))
                    db.executemany('INSERT INTO windows(map_id,start,end,tokens,split) VALUES(?,?,?,?,?)',
                                   ((mid, a, b, n, keep['split']) for a, b, n in spans))
                else:
                    db.execute('INSERT OR REPLACE INTO rejected VALUES(?,?)', (mid, error))
                completed += 1
                if completed % 100 == 0:
                    db.commit()
                    atomic_json(root / 'progress.json', dict(status='preparing', maps=number, total=total))
                    progress(f'Dataset: {number:,} / {total:,} map; minden szakasz indexelése.')
            db.executemany('INSERT OR REPLACE INTO audio_labels VALUES(?,?)', src.execute('SELECT audio_hash,payload FROM audio_labels'))
        db.commit()
        counts = dict(db.execute('SELECT split,COUNT(*) FROM maps GROUP BY split'))
        win_counts = dict(db.execute('SELECT split,COUNT(*) FROM windows GROUP BY split'))
        rejected = db.execute('SELECT COUNT(*) FROM rejected').fetchone()[0]
        if not all(counts.get(s, 0) for s in ('train', 'validation', 'test')):
            raise ValueError('Mindhárom dalonként elkülönített részhalmaz szükséges.')
        # Compact, shared read-only mmap arrays keep spawned workers small.
        for split in counts:
            rows = db.execute('SELECT id,tokens FROM windows WHERE split=? ORDER BY id', (split,)).fetchall()
            np.save(root / (split + '.npy'), np.asarray(rows, dtype=np.int64), allow_pickle=False)
        db.execute('CREATE INDEX IF NOT EXISTS window_split ON windows(split)')
        db.commit()
    finally:
        db.close()
    path.replace(root / 'index.sqlite')
    report = dict(format='lyra-dataset-1', hash=fingerprint, source_hash=manifest['hash'],
                  root=str(root), index=str(root / 'index.sqlite'), model_context=geometry,
                  tokenizer=VERSION, maps=counts, windows=win_counts, rejected=rejected,
                  split_policy='existing song-group 90/5/5; no resplit; all windows indexed',
                  complete=True, index_sha256=digest(root / 'index.sqlite'),
                  arrays_sha256={s: digest(root / (s + '.npy')) for s in counts})
    atomic_json(root / 'manifest.json', report)
    atomic_json(root / 'progress.json', dict(status='complete', **counts))
    progress(f'Lyra dataset kész: {sum(counts.values()):,} map, {sum(win_counts.values()):,} ablak; kizárt: {rejected}.')
    return report


def decode_frozen(row, cache_root):
    raw = read_bytes(row['audio_path'])
    if row.get('bytes_sha256') and hashlib.sha256(raw).hexdigest() != row['bytes_sha256']:
        raise ValueError('A rögzített audio megváltozott.')
    path, member = source_parts(row['audio_path'])
    if member is None:
        pcm = audio.decode(path)
        if audio.pcm_hash(pcm) == row['audio_hash']:
            return pcm
    with tempfile.TemporaryDirectory(prefix='decode-', dir=cache_root) as folder:
        target = Path(folder) / ('audio' + Path(member or path).suffix)
        target.write_bytes(raw)
        pcm = audio.decode(target)
    if audio.pcm_hash(pcm) != row['audio_hash']:
        # Pipe decoding can match old archive-prepared MP3 identities.
        proc = subprocess.run([audio.ffmpeg(), '-v', 'error', '-threads', '1', '-i', 'pipe:0',
                '-ac', '1', '-ar', str(audio.SR), '-f', 'f32le', 'pipe:1'], input=raw,
                capture_output=True, timeout=600, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        pcm = np.frombuffer(proc.stdout, '<f4').copy()
        if proc.returncode or audio.pcm_hash(pcm) != row['audio_hash']:
            raise ValueError('A dekódolt audio nem egyezik a rögzített datasetével.')
    return pcm


class Dataset:
    def __init__(self, manifest, split='train', config=None, cache_options=None, seed=42):
        self.manifest = load_json(manifest) if isinstance(manifest, (str, Path)) else manifest
        self.config = config or ModelConfig()
        self.split, self.seed = split, seed
        if not self.manifest or not self.manifest.get('complete'):
            raise ValueError('Előbb a teljes Lyra datasetet kell előkészíteni.')
        expected = {k: asdict(self.config)[k] for k in self.manifest['model_context']}
        if expected != self.manifest['model_context']:
            raise ValueError('A dataset és a modell kontextusbeállításai eltérnek.')
        self.cache_options = cache_options or dict(root=str(CACHE), legacy='F:/osu_cache/runtime-arrays-v1')
        self._cache = None
        self.entries = np.load(Path(self.manifest['root']) / (split + '.npy'), mmap_mode='r')
        self.tok = Tokenizer()

    def __getstate__(self):
        value = dict(self.__dict__)
        value['_cache'], value['entries'] = None, None
        return value

    def __setstate__(self, value):
        self.__dict__.update(value)
        self.entries = np.load(Path(self.manifest['root']) / (self.split + '.npy'), mmap_mode='r')

    @property
    def cache(self):
        if self._cache is None:
            from .cache import ArrayCache
            self._cache = ArrayCache(**self.cache_options)
        return self._cache

    def __len__(self):
        return len(self.entries)

    @lru_cache(maxsize=128)
    def row(self, mid):
        with readonly(self.manifest['index']) as db:
            record = db.execute('SELECT payload FROM maps WHERE id=?', (mid,)).fetchone()
        if record is None:
            raise ValueError(f'A kiválasztott map nincs a rögzített Lyra datasetben: {mid}')
        return json.loads(record[0])

    def beatmap(self, row):
        if signature(row['map_path']) != row['map_signature']:
            raise ValueError('A forrásmap megváltozott; a rögzített dataset nem írható át.')
        key = row['map_hash']
        value = self.cache.get('lyra-maps', key)
        if value is None:
            raw = read_bytes(row['map_path'], 8 * 1024 ** 2)
            if hashlib.sha256(raw).hexdigest() != key:
                raise ValueError('Map checksum eltérés.')
            value = dict(json=np.frombuffer(json.dumps(map_dict(parse_map(row['map_path']))).encode(), np.uint8).copy())
            self.cache.put('lyra-maps', key, value)
        return map_from_dict(json.loads(value['json'].tobytes()))

    def mel_window(self, row, base, count):
        if signature(row['audio_path']) != row['audio_signature']:
            raise ValueError('A forrásaudio megváltozott.')
        frames = int(math.floor(row['duration_ms'] / audio.FRAME_MS + 1e-7)) + 1
        start = round(base / audio.FRAME_MS)
        end = min(start + count, frames)
        tiles = list(range(start // 4096, (end - 1) // 4096 + 1)) if end > start else []
        keys = {i: identity([row['audio_hash'], audio.VERSION, 'mel-blocks-v1', i]) for i in tiles}
        values = {i: self.cache.get('mel-blocks', keys[i]) for i in tiles}
        if any(v is None for v in values.values()):
            with self.cache.recording_lock(row['audio_hash']):
                for i in tiles:
                    if values[i] is None:
                        values[i] = self.cache.get('mel-blocks', keys[i])
                if any(v is None for v in values.values()):
                    pcm = decode_frozen(row, self.cache.root)
                    self.cache.stats['audio_decodes'] += 1
                    if len(pcm) // audio.HOP + 1 != frames:
                        raise ValueError('A rögzített audiohossz eltér.')
                    for i in tiles:
                        if values[i] is None:
                            values[i] = dict(mel=audio.spectrogram_block(pcm, i * 4096).astype(np.float16))
                            self.cache.put('mel-blocks', keys[i], values[i])
        result = np.full((128, count), np.log(1e-8), np.float32)
        for i in tiles:
            left, right = max(start, i * 4096), min(end, (i + 1) * 4096)
            result[:, left - start:right - start] = values[i]['mel'][:, left - i * 4096:right - i * 4096]
        return (result + 5) / 7, start, frames

    def timing_targets(self, row, frames):
        key = identity([row['audio_hash'], row['timing_labels'], audio.VERSION, 'bounded-timing-v1', frames])
        value = self.cache.get('audio-targets', key)
        if value is not None:
            return value
        with readonly(self.manifest['index']) as db:
            spec = json.loads(db.execute('SELECT payload FROM audio_labels WHERE audio_hash=?', (row['audio_hash'],)).fetchone()[0])
        low, high, total = None, None, None
        invalid = np.zeros(frames, bool)
        for points in spec['timings']:
            bm = mapio.Beatmap(timing=[mapio.TimingPoint(**p) for p in points])
            target = audio.beat_targets(bm, frames)
            invalid |= audio.invalid_beat_mask(bm, frames)
            low = target.copy() if low is None else np.minimum(low, target)
            high = target.copy() if high is None else np.maximum(high, target)
            total = target.copy() if total is None else total + target
        times = np.arange(frames) * audio.FRAME_MS
        mask = (high - low < .35) & ((times >= max(0, spec['first'] - 4000)) & (times <= spec['last'] + 2000))[:, None]
        mask &= ~invalid[:, None]
        value = dict(beats=(total / len(spec['timings'])).astype(np.float16), mask=mask.astype(np.uint8))
        self.cache.put('audio-targets', key, value)
        return value

    def __getitem__(self, key):
        import torch
        import os
        index, update, number = (key, 0, 0) if isinstance(key, int) else key
        wid = int(self.entries[index, 0])
        with readonly(self.manifest['index']) as db:
            mid, start, end = db.execute('SELECT map_id,start,end FROM windows WHERE id=?', (wid,)).fetchone()
        row = self.row(mid)
        bm = self.beatmap(row)
        rng = np.random.default_rng(int(identity([self.seed, update, number, wid])[:16], 16))
        styles = row['styles'] if self.split != 'train' or rng.random() >= .2 else {}
        ctx = conditioning(bm, start, end, row['duration_ms'], row['stars'], styles, row['settings'], self.config)
        targets = [t for _, rec in self.tok.records(bm, ctx['base'], start, end) for t in rec]
        sequence = ctx['prefix'] + targets + [self.tok.ids['EOS']]
        if len(sequence) > self.config.max_tokens:
            raise ValueError('Az előkészített ablak meghaladja a tokenkeretet.')
        labels = np.asarray(sequence[1:], np.int64)
        labels[:len(ctx['prefix']) - 1] = -100
        count = math.ceil(self.config.audio_ms / audio.FRAME_MS / 4) * 4
        mel, origin, frames = self.mel_window(row, ctx['base'], count)
        full = self.timing_targets(row, frames)
        beats, mask = np.zeros((count, 2), np.float32), np.zeros((count, 2), np.float32)
        n = max(0, min(count, frames - origin))
        beats[:n], mask[:n] = full['beats'][origin:origin + n], full['mask'][origin:origin + n]
        phase = audio.phase_features(bm, origin * audio.FRAME_MS, count)
        if self.split == 'train':
            if rng.random() < .2:
                phase[:] = 0
            elif rng.random() < .25:
                phase = np.roll(phase, int(rng.integers(-3, 4)), axis=0)
        values = dict(mel=mel, beats=beats, beat_mask=mask, phase=phase,
                      tokens=np.asarray(sequence[:-1], np.int64), labels=labels,
                      summary=ctx['summary'], summary_valid=ctx['summary_valid'])
        return {**{k: torch.from_numpy(v.copy()) for k, v in values.items()},
                'meta': dict(window=wid, map=mid, start=start, end=end, update=update, number=number, worker=os.getpid()),
                'cache_stats': dict(self.cache.stats)}


def collate(samples):
    import torch
    result = {}
    for k in ('mel', 'beats', 'beat_mask', 'phase'):
        result[k] = torch.stack([s[k] for s in samples])
    for k in ('tokens', 'labels', 'summary', 'summary_valid'):
        length = max(len(s[k]) for s in samples)
        if k in ('tokens', 'labels'):
            length = math.ceil(length / 256) * 256
        out = torch.full((len(samples), length, *samples[0][k].shape[1:]), -100 if k == 'labels' else 0, dtype=samples[0][k].dtype)
        for i, sample in enumerate(samples):
            out[i, :len(sample[k])] = sample[k]
        result[k] = out
    result['meta'] = [s['meta'] for s in samples]
    result['cache_stats'] = [s['cache_stats'] for s in samples]
    result['loss_indices'] = torch.where(result['labels'].flatten() != -100)[0]
    return result


class UpdateSampler:
    def __init__(self, dataset, effective_batch, batch_size, start=0, seed=42, steps=1):
        self.dataset, self.effective, self.batch = dataset, effective_batch, batch_size
        self.start, self.seed, self.steps = start, seed, steps

    @lru_cache(maxsize=2)
    def order(self, cycle):
        return np.random.default_rng(self.seed + cycle).permutation(len(self.dataset))

    def selection(self, update):
        count = len(self.dataset)
        # All eligible windows exactly once per cycle. Bucketing only sorts
        # within a single optimizer update, retaining its original membership.
        chosen = []
        for number in range(self.effective):
            cycle, position = divmod(update * self.effective + number, count)
            i = int(self.order(cycle)[position])
            chosen.append((i, update, number))
        return sorted(chosen, key=lambda x: self.dataset.entries[x[0], 1])

    def __iter__(self):
        for update in range(self.start, self.steps):
            chosen = self.selection(update)
            for a in range(0, self.effective, self.batch):
                yield chosen[a:a + self.batch]

    def __len__(self):
        return max(0, self.steps - self.start) * (self.effective // self.batch)

import json
import sqlite3
import wave
import numpy as np
import pytest
import torch
from lyra import audio, mapio
from lyra.config import ModelConfig
from lyra.data import prepare
from lyra.io import atomic_json, digest, identity


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    index = root / 'index.sqlite'
    with sqlite3.connect(index) as db:
        db.execute('CREATE TABLE records(id TEXT PRIMARY KEY,split TEXT,group_id TEXT,payload TEXT)')
        db.execute('CREATE TABLE audio_labels(audio_hash TEXT PRIMARY KEY,payload TEXT)')
        for i, split in enumerate(('train', 'train', 'train', 'validation', 'test')):
            pcm = (np.sin(np.arange(4 * audio.SR) * (i + 1) * .07) * .2 * 32767).astype('<i2')
            wav = root / f'{i}.wav'
            with wave.open(str(wav), 'wb') as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(audio.SR); f.writeframes(pcm.tobytes())
            decoded = audio.decode(wav)
            ah = audio.pcm_hash(decoded)
            bm = mapio.Beatmap(timing=[mapio.TimingPoint(0, 500)], objects=[
                mapio.HitObject(100 + j * 25, 120 + j * 10, 500 + 250 * j) for j in range(10)])
            op = root / f'{i}.osu'
            mapio.write(bm, op)
            label = identity(i)
            row = dict(id=f'map-{i}', split=split, group=f'song-{i}', map_path=str(op), map_hash=digest(op),
                       audio_path=str(wav), audio_hash=ah, bytes_sha256=digest(wav), duration_ms=4000.,
                       stars=3., styles=dict(aim=1, streams=1, rhythm=1), settings=dict(AR=8, OD=7, CS=4, HP=5),
                       timing_labels=label, title=f'Song {i}', artist='Fixture', version='Test')
            spec = dict(timings=[[vars(p) for p in bm.timing]], first=500, last=2750)
            db.execute('INSERT INTO records VALUES(?,?,?,?)', (row['id'], split, row['group'], json.dumps(row)))
            db.execute('INSERT INTO audio_labels VALUES(?,?)', (ah, json.dumps(spec)))
    manifest = root / 'manifest.json'
    atomic_json(manifest, dict(hash='fixture-corpus', index=str(index)))
    return manifest


@pytest.fixture
def prepared(corpus, tmp_path):
    root = tmp_path / 'dataset'
    config = ModelConfig.tiny()
    prepare(corpus, root, config, workers=0, progress=lambda _: None)
    return root / 'manifest.json'

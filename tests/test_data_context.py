import json
import sqlite3
import numpy as np
import pytest
import torch
from lyra import audio, mapio
from lyra.cache import ArrayCache
from lyra.config import ModelConfig
from lyra.context import conditioning, windows
from lyra.data import Dataset, UpdateSampler, collate, prepare
from lyra.io import load_json, readonly
from lyra.tokenizer import Tokenizer


def test_all_maps_and_all_windows_keep_original_song_splits(prepared, corpus):
    manifest = load_json(prepared)
    assert manifest['maps'] == dict(train=3, validation=1, test=1)
    assert manifest['complete']
    assert manifest['source_hash'] == 'fixture-corpus'
    with readonly(manifest['index']) as db:
        assert db.execute('SELECT COUNT(*) FROM maps').fetchone()[0] == 5
        assert not db.execute('SELECT group_id FROM maps GROUP BY group_id HAVING COUNT(DISTINCT split)>1').fetchall()
    assert prepare(corpus, prepared.parent, ModelConfig.tiny(), workers=0)['hash'] == manifest['hash']


def test_dense_windows_cover_every_event_without_context_loss():
    config = ModelConfig.tiny()
    bm = mapio.Beatmap(timing=[mapio.TimingPoint(0, 500)], objects=[mapio.HitObject(200, 200, t) for t in range(0, 70000, 25)])
    spans = windows(bm, 72000, config)
    assert spans[0][0] == 0 and spans[-1][1] == 72000
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert sum(sum(a <= o.time < b for o in bm.objects) for a, b, _ in spans) == len(bm.objects)
    tok = Tokenizer()
    for start, end, _ in spans:
        ctx = conditioning(bm, start, end, 72000, 5, {}, {}, config)
        target = [t for _, rec in tok.records(bm, ctx['base'], start, end) for t in rec]
        assert len(ctx['prefix']) + len(target) + 1 <= config.max_tokens


def test_context_is_identical_for_training_and_generation_and_has_no_future(prepared, tmp_path):
    config = ModelConfig.tiny()
    ds = Dataset(prepared, 'validation', config, dict(root=str(tmp_path / 'cache'), gb=.05, ram_gib=.02, min_free_gib=0))
    sample = ds[0]
    row = ds.row(sample['meta']['map'])
    bm = ds.beatmap(row)
    ctx = conditioning(bm, sample['meta']['start'], sample['meta']['end'], row['duration_ms'], row['stars'], row['styles'], row['settings'], config)
    assert sample['tokens'][:len(ctx['prefix'])].tolist() == ctx['prefix']
    bm = mapio.Beatmap(timing=[mapio.TimingPoint(0, 500)], objects=[mapio.HitObject(100, 100, t) for t in (1000, 10000, 50000, 60000)])
    a = conditioning(bm, 50000, 70000, 90000, 5, {}, {}, config)
    bm.objects[-1].x = 400
    b = conditioning(bm, 50000, 70000, 90000, 5, {}, {}, config)
    np.testing.assert_array_equal(a['summary'], b['summary'])
    assert a['prefix'] == b['prefix']


def test_active_slider_survives_context_boundary():
    config = ModelConfig()
    bm = mapio.Beatmap(timing=[mapio.TimingPoint(0, 500)], objects=[
        mapio.HitObject(100, 100, 1000, 'slider', points=[(400, 100)], length=14000)])
    ctx = conditioning(bm, 50000, 74000, 90000, 5, {}, {}, config)
    assert Tokenizer().ids['SLIDER'] in ctx['prefix']


def test_sampler_covers_all_windows_and_resume_preserves_membership(prepared, tmp_path):
    ds = Dataset(prepared, config=ModelConfig.tiny(), cache_options=dict(root=str(tmp_path / 'c'), gb=.05, ram_gib=.01, min_free_gib=0))
    sampler = UpdateSampler(ds, 2, 1, steps=8)
    first = [v[0] for u in range(2) for v in sampler.selection(u)]
    assert set(first) == set(range(len(ds)))
    resumed = UpdateSampler(ds, 2, 2, start=3, steps=8)
    assert sampler.selection(3) == resumed.selection(3)


def test_cache_warm_path_identical_and_source_changes_rejected(prepared, tmp_path):
    ds = Dataset(prepared, config=ModelConfig.tiny(), cache_options=dict(root=str(tmp_path / 'c'), gb=.05, ram_gib=.01, min_free_gib=0))
    a, b = ds[0], ds[0]
    torch.testing.assert_close(a['mel'], b['mel'], atol=0, rtol=0)
    assert ds.cache.stats['audio_decodes'] == 1
    assert ds.cache.stats['ram_hits'] >= 3
    assert collate([a, b])['labels'].shape[0] == 2
    row = ds.row(a['meta']['map'])
    with open(row['map_path'], 'a', encoding='utf-8') as f:
        f.write('\n// changed\n')
    with pytest.raises(ValueError, match='megváltozott'):
        ds[0]


def test_cached_training_audio_matches_generation_frame_grid(prepared, tmp_path):
    ds = Dataset(prepared, 'validation', ModelConfig.tiny(), dict(root=str(tmp_path / 'c'), gb=.05, ram_gib=.01, min_free_gib=0))
    sample = ds[0]
    row = ds.row(sample['meta']['map'])
    full = audio.spectrogram(audio.decode(row['audio_path'])).astype(np.float16).astype(np.float32)
    generated = audio.window(full, 0, ds.config.audio_ms)
    np.testing.assert_array_equal(generated, sample['mel'].numpy())


def test_corrupt_or_full_cache_never_removes_dataset_members(tmp_path):
    cache = ArrayCache(tmp_path / 'cache', gb=.00001, ram_gib=0, min_free_gib=0)
    value = dict(a=np.arange(10, dtype=np.float32))
    cache.put('x', 'first', value)
    path = next(cache.root.glob('entry-*.npz'))
    path.write_bytes(b'broken')
    assert cache.get('x', 'first') is None
    cache.put('x', 'large', dict(a=np.zeros(100000)))
    assert cache.stats['bypasses'] == 1
    assert cache.get('x', 'large') is None


def test_prepare_rejects_song_leakage(corpus, tmp_path):
    manifest = load_json(corpus)
    with sqlite3.connect(manifest['index']) as db:
        payload = json.loads(db.execute("SELECT payload FROM records WHERE id='map-3'").fetchone()[0])
        payload['group'] = 'song-0'
        db.execute("UPDATE records SET payload=? WHERE id='map-3'", (json.dumps(payload),))
    with pytest.raises(ValueError, match='dalátfedés'):
        prepare(corpus, tmp_path / 'bad', ModelConfig.tiny(), workers=0)

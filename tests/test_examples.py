import json
from pathlib import Path
import zipfile
import numpy as np
import pytest
import torch
from lyra import audio, examples, mapio
from lyra.generation import training_sample
from lyra.io import load_json


def test_external_excerpt_is_frozen_and_never_reads_a_reference_map(corpus, tmp_path, monkeypatch):
    import lyra.generation as generation
    source = corpus.parent / '0.wav'
    manifest = examples.prepare_example(source, tmp_path / 'example', 1000, 3000)
    spec = examples.load_example(manifest)
    clip = audio.decode(spec['audio_path'])
    expected = audio.decode(source)[audio.SR:3 * audio.SR]
    assert len(clip) == pytest.approx(len(expected), abs=audio.SR * .05)
    assert np.corrcoef(clip[:len(expected)], expected[:len(clip)])[0, 1] > .98
    assert spec['source_range_ms'] == [1000, 3000]
    before = Path(spec['audio_path']).stat().st_mtime_ns
    assert examples.prepare_example(source, tmp_path / 'example', 1000, 3000) == manifest
    assert Path(spec['audio_path']).stat().st_mtime_ns == before
    with pytest.raises(ValueError):
        examples.prepare_example(source, tmp_path / 'beyond', 1000, 5000)
    source.unlink()  # Preview survives removal of the original desktop file.
    monkeypatch.setattr(examples, 'CACHE', tmp_path / 'cache')
    cache_type = examples.ArrayCache
    def small_cache(**kwargs):
        return cache_type(**{**kwargs, 'gb': .05, 'min_free_gib': 0})
    monkeypatch.setattr(examples, 'ArrayCache', small_cache)
    seen, decodes = [], []
    decode = audio.decode
    def counted_decode(path):
        decodes.append(path)
        return decode(path)
    monkeypatch.setattr(audio, 'decode', counted_decode)
    def timing(model, mel, duration, device, cfg, cancelled):
        assert not model.training and cfg.timing_map is None
        assert duration == pytest.approx(2000, abs=50)
        return [mapio.TimingPoint(0, 500)], dict(source='model-audio', confidence=.8)
    def candidate(model, mel, duration, timing, cfg, settings, device, seed, progress, cancelled):
        seen.append((mel.copy(), seed, cfg.temperature, cfg.stars))
        return mapio.Beatmap(timing=timing, objects=[mapio.HitObject(100, 100, 500),
                                                  mapio.HitObject(200, 200, 1000)])
    monkeypatch.setattr(generation, 'infer_timing', timing)
    monkeypatch.setattr(generation, 'candidate', candidate)
    model = torch.nn.Linear(1, 1).train()
    for step in (1000, 2000):
        output = training_sample(model, object(), tmp_path / f'step-{step}', torch.device('cpu'),
                                 audio_example=manifest, reference_map_id='must-never-be-used', training_step=step)
        assert model.training
        with zipfile.ZipFile(output) as z:
            meta = json.loads(z.read('generation.json'))
            restored = mapio.parse(z.read('Lyra.osu').decode())
        assert meta['example_source'] == 'external_audio' and meta['reference_map_id'] is None
        assert not meta['reference_timing'] and meta['timing']['source'] == 'model-audio'
        assert meta['source_range_ms'] == [1000, 3000]
        assert meta['training_step'] == step
        assert meta['complete_excerpt'] and not meta['complete_song']
        assert restored.metadata['Title'] == spec['title']
        assert f'step {step}' in restored.metadata['Version']
    assert len(decodes) == 1
    np.testing.assert_array_equal(seen[0][0], seen[1][0])
    assert seen[0][1:] == seen[1][1:] == (2026, .6, 6.5)
    Path(spec['audio_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='megváltozott'):
        examples.load_example(manifest)
    assert load_json(manifest)['source_path'] == str(source)

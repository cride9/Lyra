import zipfile
from dataclasses import replace
import numpy as np
import pytest
import torch
from lyra import mapio
from lyra.config import ModelConfig
from lyra.data import Dataset
from lyra.generation import GenerationConfig, candidate, export, difficulty
from lyra.model import Lyra
from lyra.tokenizer import Tokenizer, Grammar
from lyra.geometry import radius, validate_map, validate_object


@pytest.mark.parametrize('cs', [0, 4, 7, 10])
@pytest.mark.parametrize('kind', ['CIRCLE', 'SLIDER'])
def test_generation_grammar_keeps_complete_hit_radius_in_playfield(cs, kind):
    tok = Tokenizer()
    grammar = Grammar(tok, 0, 0, 3000, 3000, circle_size=cs)
    for token in [tok.ids[kind], *tok.time(500, 0)]:
        grammar.consume(token)
    margin = radius(cs) + (0.25 if kind == 'SLIDER' else 0)
    for extent in (512, 384):
        coordinates = [(tok.v('XY', t) - 1024) * 2 for t in grammar.allowed()]
        assert min(coordinates) >= margin
        assert max(coordinates) <= extent - margin
        grammar.consume(grammar.allowed()[0])


def _geometry_retry_example(monkeypatch, bad_y_values):
    import lyra.generation as generation
    tok = Tokenizer()
    reference = mapio.Beatmap(timing=[mapio.TimingPoint(0, 500)])
    circle = mapio.HitObject(100, 100, 500)
    slider = mapio.HitObject(100, 100, 1500, 'slider', curve='L',
                            points=[(200, bad_y_values[0])], length=280)
    first = tok.object(reference, circle, 0)
    record = tok.object(reference, slider, 0)
    plan = first + record + [tok.ids['EOS']]
    anchor_y_index = len(plan) - 4
    position, rejected = 0, []

    def choose(logits, allowed, cfg, generator, bias=None):
        nonlocal position
        if position == anchor_y_index:
            preferred = [tok.xy(0, y)[1] for y in bad_y_values] + [tok.xy(0, 100)[1]]
            token = next(t for t in preferred if t in allowed)
        else:
            token = plan[position]
        assert token in allowed
        position += 1
        return token

    def check(obj, cs):
        nonlocal position
        try:
            validate_object(obj, cs)
        except ValueError:
            rejected.append(obj)
            position = len(first)
            raise

    monkeypatch.setattr(generation, 'sample_token', choose)
    monkeypatch.setattr(generation, 'validate_object', check)
    config = replace(ModelConfig.tiny(), audio_ms=4000, history_ms=1000, output_ms=3000)
    model = Lyra(len(tok), config).eval()
    return model, reference, rejected


def test_geometry_retry_changes_greedy_choice_and_keeps_completed_objects(monkeypatch):
    model, reference, rejected = _geometry_retry_example(monkeypatch, [-100])
    bm = candidate(model, np.zeros((128, 401), np.float32), 3000, reference.timing,
                   GenerationConfig(temperature=0), dict(AR=8, OD=7, CS=4, HP=5),
                   torch.device('cpu'), 42, lambda _: None)
    validate_map(bm, 3000)
    assert len(rejected) == 1
    assert [(o.kind, o.time) for o in bm.objects] == [('circle', 500), ('slider', 1500)]
    assert bm.objects[0].x == bm.objects[0].y == 100
    assert bm.objects[1].points == [(200, 100)]


def test_geometry_retry_limit_reports_actual_object_and_validation_reason(monkeypatch):
    import lyra.generation as generation
    monkeypatch.setattr(generation, '_MAX_OBJECT_RETRIES', 2)
    model, reference, rejected = _geometry_retry_example(monkeypatch, [-100, -98, -96])
    with pytest.raises(ValueError, match=r'slider at 1500 ms.*2 retries.*Out-of-bounds slider'):
        candidate(model, np.zeros((128, 401), np.float32), 3000, reference.timing,
                  GenerationConfig(temperature=0), dict(AR=8, OD=7, CS=4, HP=5),
                  torch.device('cpu'), 42, lambda _: None)
    assert len(rejected) == 3


def test_real_inference_and_grammar_generate_across_sections(monkeypatch):
    import lyra.generation as generation
    tok = Tokenizer()
    reference = mapio.Beatmap(timing=[mapio.TimingPoint(0, 500)])
    objects = [mapio.HitObject(100, 100, 500), mapio.HitObject(200, 200, 1500)]
    planned = iter([*tok.object(reference, objects[0], 0), tok.ids['EOS'],
                    *tok.object(reference, objects[1], 0), tok.ids['EOS']])
    def choose(logits, allowed, cfg, generator, bias=None):
        assert logits.shape == (len(tok),) and torch.isfinite(logits).all()
        token = next(planned)
        assert token in allowed
        return token
    # Only sampling is controlled; the actual audio encoder, cached decoder,
    # grammar, context continuation, decoding and geometry validation all run.
    monkeypatch.setattr(generation, 'sample_token', choose)
    config = replace(ModelConfig.tiny(), audio_ms=2000, history_ms=1000, output_ms=1000)
    model = Lyra(len(tok), config).eval()
    bm = candidate(model, np.zeros((128, 201), np.float32), 2000, reference.timing,
                   GenerationConfig(), dict(AR=8, OD=7, CS=4, HP=5), torch.device('cpu'),
                   2026, lambda _: None)
    assert [(o.time, o.x, o.y) for o in bm.objects] == [(500, 100, 100), (1500, 200, 200)]
    assert next(planned, None) is None


def test_full_song_continuation_keeps_previous_objects_and_crossing_slider(monkeypatch):
    import lyra.generation as generation
    seen = []
    def section(model, mel, bm, start, end, duration, cfg, settings, device, rng, cancelled):
        seen.append((start, len(bm.objects)))
        if start == 0:
            bm.objects.append(mapio.HitObject(100, 100, 23000, 'slider', points=[(380, 100)], length=280, repeats=3))
        else:
            busy = bm.objects[-1].time + bm.duration(bm.objects[-1])
            bm.objects.append(mapio.HitObject(200, 200, round(max(start, busy + 1))))
        return end
    monkeypatch.setattr(generation, 'section', section)
    model = Lyra(len(Tokenizer()), ModelConfig.tiny())
    bm = candidate(model, np.zeros((128, 6000), np.float32), 60000, [mapio.TimingPoint(0, 500)],
                   GenerationConfig(), dict(AR=8, OD=7, CS=4, HP=5), torch.device('cpu'), 42, lambda _: None)
    assert seen == [(0, 0), (24000, 1), (48000, 2)]
    assert bm.objects[1].time > 25000


@pytest.mark.parametrize('training_step', [None, 1000])
def test_export_is_osu_readable_and_leaves_one_archive(corpus, tmp_path, training_step):
    wav = corpus.parent / '0.wav'
    bm = mapio.Beatmap(timing=[mapio.TimingPoint(0, 500)], objects=[mapio.HitObject(100, 100, 500), mapio.HitObject(200, 200, 1000)])
    stars = difficulty(bm)['stars']
    metadata = dict(measured_stars=stars)
    if training_step is not None:
        metadata.update(training_step=training_step, reference_title='Hakai')
    target = export(bm, wav, tmp_path / 'output', metadata)
    assert list((tmp_path / 'output').iterdir()) == [__import__('pathlib').Path(target)]
    with zipfile.ZipFile(target) as z:
        restored = mapio.parse(z.read('Lyra.osu').decode())
        assert restored.metadata['Creator'] == 'Lyra'
        if training_step is not None:
            assert restored.metadata['Title'] == 'Hakai'
            assert f'step {training_step}' in restored.metadata['Version']
            assert f'step {training_step}' in target
        assert 'audio.ogg' in z.namelist()
        assert difficulty(restored)['stars'] == pytest.approx(stars)


def test_generation_controls():
    assert GenerationConfig(preset='Streams').styles()['streams'] == 2
    with pytest.raises(ValueError):
        GenerationConfig(temperature=-1).validate()


@pytest.mark.parametrize('reference_id,expected_id,expected_split', [
    ('map-0', 'map-0', 'train'), (None, 'map-3', 'validation')])
def test_periodic_example_uses_fixed_reference_for_full_song(prepared, tmp_path, monkeypatch,
                                                             reference_id, expected_id, expected_split):
    import lyra.generation as generation
    dataset = Dataset(prepared, 'validation', ModelConfig.tiny(),
                      dict(root=str(tmp_path / 'cache'), gb=.05, ram_gib=.01, min_free_gib=0, legacy=None))
    model = Lyra(len(Tokenizer()), ModelConfig.tiny())
    calls, exports = [], []
    def fake_candidate(model, mel, duration, timing, cfg, settings, device, seed, progress, cancelled,
                       span=None):
        assert not model.training
        calls.append((mel.copy(), duration, seed, cfg.temperature, dict(settings), span))
        return mapio.Beatmap(timing=timing, objects=[mapio.HitObject(100, 100, 500)])
    def fake_export(bm, source, destination, metadata):
        exports.append((source, bm.metadata.copy(), metadata))
        return str(destination / 'example.osz')
    monkeypatch.setattr(generation, 'candidate', fake_candidate)
    monkeypatch.setattr(generation, 'difficulty', lambda _: dict(stars=2.5))
    monkeypatch.setattr(generation, 'export', fake_export)
    for step in (1000, 2000):
        destination = tmp_path / 'run' / 'samples' / f'step-{step}'
        generation.training_sample(model, dataset, destination, torch.device('cpu'),
                                   reference_map_id=reference_id, training_step=step)
        assert model.training
        metadata = exports[-1][2]
        assert metadata['reference_map_id'] == expected_id
        assert metadata['reference_split'] == expected_split
        assert metadata['training_step'] == step
        assert metadata['complete_song'] and metadata['reference_timing']
        assert metadata['requested_stars'] == 3
        assert metadata['seed'] == metadata['settings']['seed'] == 2026
        assert exports[-1][0] == __import__('pathlib').Path(dataset.row(expected_id)['audio_path'])
        assert exports[-1][1]['Title'] == dataset.row(expected_id)['title']
    np.testing.assert_array_equal(calls[0][0], calls[1][0])
    assert calls[0][1:] == calls[1][1:] == (4000., 2026, .6, dict(AR=8, OD=7, CS=4, HP=5), None)
    assert dataset.split == 'validation' and len(dataset) == 1
    with pytest.raises(ValueError, match='frozen Lyra dataset'):
        generation.training_sample(model, dataset, tmp_path, torch.device('cpu'), reference_map_id='missing')

    def rejected(*args, **kwargs):
        raise ValueError('invalid early example')
    monkeypatch.setattr(generation, 'candidate', rejected)
    with pytest.raises(ValueError, match='invalid early example'):
        generation.training_sample(model, dataset, tmp_path, torch.device('cpu'), reference_map_id=reference_id)
    assert model.training

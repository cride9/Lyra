from types import SimpleNamespace
import numpy as np
import pytest
import torch
from lyra import audio, mapio
from lyra.config import ModelConfig
from lyra.generation import GenerationConfig, infer_timing
from lyra.stitching import replay_map


def test_replay_preserves_boundaries_clocks_crossing_objects_and_breaks():
    source = mapio.Beatmap(timing=[mapio.TimingPoint(0, 500), mapio.TimingPoint(23000, -50, uninherited=False),
                                   mapio.TimingPoint(48000, 400), mapio.TimingPoint(48000, -80, uninherited=False)],
        objects=[mapio.HitObject(100, 100, t) for t in (0, 23999, 24000, 24001, 47999, 48000, 64000, 72000)],
        breaks=[(25000, 27000), (69000, 70000)])
    source.objects += [mapio.HitObject(150, 150, 23800, 'slider', curve='L', points=[(300, 150)], length=560),
                       mapio.HitObject(200, 200, 46000, 'spinner', end_time=53000)]
    source.objects.sort(key=lambda o: o.time)
    restored, report = replay_map(source, 73555)
    assert report['passed'], report['first_difference']
    assert report['start_time_mismatches'] == 0
    assert len(restored.objects) == len(source.objects)
    assert restored.breaks == source.breaks
    assert report['grammar_rejections']  # Deliberate overlapping source notes are diagnosed, not dropped.
    assert report['sections'][-1]['end'] == 73555
    assert report['sections'][1]['before']['objects'] > 0


def test_replay_token_budget_shortens_only_between_complete_time_groups():
    source = mapio.Beatmap(timing=[mapio.TimingPoint(0, 500)], objects=[
        mapio.HitObject(100 + j * 40, 150, at) for at in range(0, 70001, 50) for j in range(2)])
    restored, report = replay_map(source, 71555, ModelConfig.tiny())
    assert report['passed'], report['first_difference']
    assert len(restored.objects) == len(source.objects)
    assert any(s['shortened'] for s in report['sections'])
    assert len(report['sections']) > 3


def test_replay_detects_a_production_assembly_time_shift(monkeypatch):
    import lyra.generation as generation
    original = generation.commit_event
    def faulty(*args, **kwargs):
        original(*args, **kwargs)
        args[0].objects[-1].time += 1
    monkeypatch.setattr(generation, 'commit_event', faulty)
    source = mapio.Beatmap(timing=[mapio.TimingPoint(0, 500)], objects=[mapio.HitObject(100, 100, 500)])
    _, report = replay_map(source, 1000)
    assert not report['passed']
    assert report['first_difference']['expected'][0] == 500
    assert report['first_difference']['actual'][0] == 501
    assert report['first_difference']['section']['base'] == 0
    assert report['source_quantization']['max_end_error_ms'] == 0


def test_replay_also_detects_changed_timing_points_with_unchanged_note_tokens(monkeypatch):
    import lyra.generation as generation
    original = generation.commit_event
    def faulty(*args, **kwargs):
        original(*args, **kwargs)
        args[0].timing[0].time += 1
    monkeypatch.setattr(generation, 'commit_event', faulty)
    source = mapio.Beatmap(timing=[mapio.TimingPoint(0, 500)], objects=[mapio.HitObject(100, 100, 500)])
    _, report = replay_map(source, 1000)
    assert not report['passed']
    assert report['first_difference']['kind'] == 'timing'
    assert report['first_difference']['expected']['time'] == 0
    assert report['first_difference']['actual']['time'] == 1


@pytest.mark.parametrize('duration', [264000, 73555])
def test_timing_head_stitching_covers_all_global_frames_without_shifts(monkeypatch, duration):
    import lyra.generation as generation
    frames = int(duration / audio.FRAME_MS) + 1
    mel = np.broadcast_to(np.linspace(-1, 1, frames, dtype=np.float32), (128, frames)).copy()
    def encode(x):
        logits = x[:, 0] * 7 - 5
        return None, None, torch.stack((logits, logits), -1)
    def inspect(scores, **kwargs):
        expected = torch.from_numpy(mel[0]).sigmoid().numpy()
        np.testing.assert_allclose(scores[:, 0], expected, atol=2e-7, rtol=1e-6)
        np.testing.assert_array_equal(scores[:, 0], scores[:, 1])
        return [], dict(verified=True)
    monkeypatch.setattr(generation, 'estimate', inspect)
    model = SimpleNamespace(config=ModelConfig(), encode=encode)
    _, report = infer_timing(model, mel, duration, torch.device('cpu'), GenerationConfig())
    assert report['verified']

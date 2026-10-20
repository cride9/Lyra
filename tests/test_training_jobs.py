import os
from dataclasses import replace
import csv
import io
import json
from pathlib import Path
import time
import psutil
import torch
from fastapi.testclient import TestClient
from lyra import jobs
from lyra.config import TrainConfig
from lyra.io import atomic_json, load_json
from lyra.metrics import Metrics
from lyra.runtime import load_checkpoint
from lyra.server import create_app
from lyra.training import train


def test_training_resume_restores_optimizer_scheduler_rng_and_next_window(prepared, tmp_path):
    cfg = TrainConfig(model='tiny', device='cpu', batch_size=1, effective_batch=2, workers=0,
                      ram_cache_gib=.05, disk_cache_gb=.05, min_free_gib=0, warmup=2,
                      steps=3, validate_every=2, validation_windows=1, sample_every=0)
    uninterrupted = tmp_path / 'full'
    partial = tmp_path / 'partial'
    train(prepared, uninterrupted, cfg, progress=lambda _: None)
    train(prepared, partial, cfg, limit_updates=1, progress=lambda _: None)
    from lyra.examples import prepare_example
    source_audio = prepared.parent.parent / 'source/0.wav'
    external = prepare_example(source_audio, tmp_path / 'external', 500, 3500)
    resumed = replace(cfg, sample_every=1000, sample_map_id='map-0', sample_audio_example=external)
    train(prepared, partial, resumed, resume=partial / 'last.pt', progress=lambda _: None)
    a, b = load_checkpoint(uninterrupted / 'last.pt'), load_checkpoint(partial / 'last.pt')
    assert a['step'] == b['step'] == 3
    assert a['scheduler'] == b['scheduler']
    assert b['train_config']['sample_every'] == 1000
    assert b['train_config']['sample_map_id'] == 'map-0'
    assert b['train_config']['sample_audio_example'] == external
    for name in a['model']:
        torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
    assert a['coverage']['visited'] == 6
    assert len(Metrics(partial).history()) == 3
    assert Metrics(uninterrupted).latest()['loss_ema'] == Metrics(partial).latest()['loss_ema']


def test_metrics_resume_removes_abandoned_suffix(tmp_path):
    metrics = Metrics(tmp_path)
    for i in range(1, 6):
        metrics.write(dict(step=i, loss=1 / i))
    metrics.truncate_after(3)
    assert [r['step'] for r in metrics.history()] == [1, 2, 3]
    audit = list(metrics.records(include_archived=True))
    assert len(audit) == 5
    assert {v['step'] for v in audit if v['record_type'] == 'rolled_back_update'} == {4, 5}


def test_preview_type_error_preserves_training_and_resume_recovers_missing_example(prepared, tmp_path, monkeypatch):
    import random
    import numpy as np
    import lyra.generation as generation
    cfg = TrainConfig(model='tiny', device='cpu', batch_size=1, effective_batch=2, workers=0,
                      ram_cache_gib=.05, disk_cache_gb=.05, min_free_gib=0, warmup=2,
                      steps=2, validate_every=100, validation_windows=1,
                      sample_every=1, sample_map_id='map-0')
    attempts = []
    def broken_preview(model, dataset, destination, device, cancelled, **kwargs):
        attempts.append(kwargs['training_step'])
        random.random(); np.random.random(); torch.rand(3)
        model.eval()
        raise TypeError('preview interface mismatch')
    monkeypatch.setattr(generation, 'training_sample', broken_preview)
    run = tmp_path / 'preview'
    baseline = tmp_path / 'baseline'
    train(prepared, run, cfg, limit_updates=1, progress=lambda _: None)
    rejected = run / 'samples/step-1/rejected.json'
    assert load_json(rejected)['error_type'] == 'TypeError'
    # Simulate the original crash: checkpoint exists but its preview did not finish.
    rejected.unlink()
    train(prepared, run, cfg, resume=run / 'last.pt', progress=lambda _: None)
    assert attempts == [1, 1, 2]
    assert load_json(run / 'status.json')['status'] == 'complete'
    assert len(Metrics(run).history()) == 2
    for step in (1, 2):
        failure = load_json(run / f'samples/step-{step}/rejected.json')
        assert failure['error_type'] == 'TypeError'
        assert 'preview interface mismatch' in failure['traceback']
    train(prepared, baseline, replace(cfg, sample_every=0), progress=lambda _: None)
    actual, expected = load_checkpoint(run / 'last.pt'), load_checkpoint(baseline / 'last.pt')
    assert actual['step'] == expected['step'] == 2
    assert actual['scheduler'] == expected['scheduler']
    for name in expected['model']:
        torch.testing.assert_close(actual['model'][name], expected['model'][name], rtol=0, atol=0)


def test_stop_during_preview_saves_completed_update(prepared, tmp_path, monkeypatch):
    import lyra.generation as generation
    cfg = TrainConfig(model='tiny', device='cpu', batch_size=1, effective_batch=2, workers=0,
                      ram_cache_gib=.05, disk_cache_gb=.05, min_free_gib=0, warmup=2,
                      steps=2, validate_every=100, validation_windows=1, sample_every=1)
    def stopped_preview(*args, **kwargs):
        raise InterruptedError('stop during preview')
    monkeypatch.setattr(generation, 'training_sample', stopped_preview)
    run = tmp_path / 'stopped'
    train(prepared, run, cfg, progress=lambda _: None)
    assert load_json(run / 'status.json')['status'] == 'cancelled'
    assert load_checkpoint(run / 'last.pt')['step'] == 1
    assert not (run / 'samples/step-1/rejected.json').exists()


def test_metrics_exports_full_history_with_nested_fields_and_rollback_archive(tmp_path):
    run = tmp_path / 'runs' / 'main'
    metrics = Metrics(run)
    for step in range(1, 4):
        metrics.write(dict(step=step, loss=1 / step, seconds_per_update=10 + step,
                           cache=dict(ram_hits=step), validation=dict(beat_loss=.1),
                           timestamp_utc=f'2026-10-19T10:00:0{step}+00:00'))
    metrics.truncate_after(2)
    metrics.write(dict(step=3, loss=.5, timestamp_utc='2026-10-19T10:01:00+00:00'))
    with TestClient(create_app(tmp_path)) as client:
        response = client.get('/api/metrics/export?run=main&format=csv')
        assert response.status_code == 200
        assert 'attachment' in response.headers['content-disposition']
        rows = list(csv.DictReader(io.StringIO(response.content.decode('utf-8-sig'))))
        assert len(rows) == 3 and rows[0]['cache.ram_hits'] == '1'
        assert rows[0]['validation.beat_loss'] == '0.1'
        audit = client.get('/api/metrics/export?run=main&format=jsonl&include_archived=true')
        values = [json.loads(line) for line in audit.text.splitlines()]
        assert len(values) == 4
        assert sum(v['record_type'] == 'rolled_back_update' for v in values) == 1
        assert client.get('/api/metrics/export?format=bad').status_code == 400
    assert metrics.recent_speed() == 11.5


def test_refresh_and_new_server_reconnect_to_same_worker_and_loss(tmp_path):
    spec = tmp_path / 'jobs' / 'live.json'
    lease = dict(pid=os.getpid(), created=psutil.Process().create_time())
    atomic_json(spec, dict(action='train', arguments={}, status='running', **lease))
    spec.with_suffix('.status.json').write_bytes(b'\x00\x00')
    run = tmp_path / 'runs' / 'main'
    atomic_json(run / 'status.json', dict(step=123, loss=1.2, status='running'))
    Metrics(run).write(dict(step=123, loss=1.2))
    for _ in range(2):
        with TestClient(create_app(tmp_path)) as client:
            for _ in range(2):
                state = client.get('/api/state').json()
                assert state['job']['job_id'] == 'live'
                assert state['job']['status'] == 'running'
                assert state['run']['step'] == 123
                assert state['history'][0]['loss'] == 1.2
            assert client.get('/').status_code == 200
            assert client.get('/static/app.js').status_code == 200
            assert client.post('/api/stop').status_code == 200
    assert spec.with_suffix('.cancel').exists()


def test_pid_reuse_does_not_attach_to_unrelated_process(tmp_path):
    spec = tmp_path / 'jobs' / 'old.json'
    atomic_json(spec, dict(action='train', status='running', pid=os.getpid(), created=1))
    assert jobs.status(spec)['status'] == 'interrupted'


def test_detached_prepare_worker_publishes_durable_result(corpus, tmp_path):
    home = tmp_path / 'job-home'
    path = Path(jobs.start(home, 'prepare', dict(source=str(corpus), destination=str(home / 'dataset'), workers=0)))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = jobs.status(path)
        if state['status'] not in ('starting', 'running'):
            break
        time.sleep(.1)
    assert state['status'] == 'complete', state
    assert load_json(home / 'dataset' / 'manifest.json')['complete']
    assert TestClient(create_app(home)).get('/api/state').json()['job']['result']['complete']

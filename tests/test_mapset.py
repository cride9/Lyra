from dataclasses import asdict
import json
import zipfile
import pytest
import torch
from fastapi.testclient import TestClient
from lyra import audio, mapio, mapset, jobs
from lyra.config import ModelConfig
from lyra.generation import GenerationConfig
from lyra.io import load_json
from lyra.server import create_app


@pytest.mark.parametrize('stars,name', [(0,'Easy'),(1.9999,'Easy'),(2,'Normal'),(2.6999,'Normal'),
    (2.7,'Hard'),(3.9999,'Hard'),(4,'Insane'),(5.2999,'Insane'),(5.3,'Expert'),(6.4999,'Expert'),
    (6.5,'Expert+'),(25,'Expert+')])
def test_bands_are_contiguous_and_use_unrounded_stars(stars, name):
    assert [t.name for t in mapset.TIERS if t.contains(stars)] == [name]


@pytest.mark.parametrize('targets', [{'Easy':2}, {'Normal':1.99}, {'Expert+':6.49}, {'Expert+':21},
                                    {'Hard':float('nan')}, {'oops':3}, []])
def test_invalid_targets_are_rejected(targets):
    with pytest.raises(ValueError):
        mapset.validate_targets(targets)


def test_profiles_and_explicit_zero_overrides():
    assert [(mapset.profile(t,GenerationConfig())['AR'],mapset.profile(t,GenerationConfig())['OD'])
            for t in mapset.TIERS] == [(3,3),(5,4),(7,6),(8.5,7.5),(9.2,8.5),(9.7,9)]
    for tier in mapset.TIERS:
        assert mapset.profile(tier,GenerationConfig(ar=0,od=2,cs=0,hp=10)) == dict(AR=0,OD=2,CS=0,HP=10)


def fake_pipeline(monkeypatch, tmp_path, failures=()):
    model = torch.nn.Linear(1,1)
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'fixture-checkpoint')
    calls = dict(loads=0,decode=0,mel=0,timing=0,candidates=[])
    def factory(*args):
        calls['loads'] += 1
        return model
    monkeypatch.setattr(mapset,'Lyra',factory)
    monkeypatch.setattr(mapset,'load_checkpoint',lambda _:dict(model_config=asdict(ModelConfig.tiny()),
                          model=model.state_dict(),step=50304))
    decode,spectrogram=audio.decode,audio.spectrogram
    def read(path):
        calls['decode']+=1
        return decode(path)
    def mel(pcm):
        calls['mel']+=1
        return spectrogram(pcm)
    monkeypatch.setattr(audio,'decode',read)
    monkeypatch.setattr(audio,'spectrogram',mel)
    def timing(*args):
        calls['timing']+=1
        return [mapio.TimingPoint(0,500)],dict(confidence=.9)
    monkeypatch.setattr(mapset,'infer_timing',timing)
    def candidate(model,mel,duration,timing,cfg,settings,device,seed,progress,cancelled):
        calls['candidates'].append((cfg.stars,seed,settings,timing))
        if cfg.stars in failures:
            raise ValueError('deliberately invalid candidate')
        # Hard deliberately misses its band; Easy has two in-band choices.
        stars=({42:2.1,43:1.8,44:1.4}.get(seed,cfg.stars) if cfg.stars==1.5
               else cfg.stars+.8 if cfg.stars==3.3 else cfg.stars)
        bm=mapio.Beatmap(timing=list(timing),objects=[mapio.HitObject(100,100,500),mapio.HitObject(200,200,1000)])
        bm.metadata['FixtureStars']=str(stars)
        for key,value in settings.items():
            bm.difficulty[dict(AR='ApproachRate',OD='OverallDifficulty',CS='CircleSize',HP='HPDrainRate')[key]]=str(value)
        return bm
    monkeypatch.setattr(mapset,'candidate',candidate)
    monkeypatch.setattr(mapset,'difficulty',lambda bm:dict(stars=float(bm.metadata['FixtureStars'])))
    return checkpoint,calls


def test_mapset_shares_audio_model_and_timing_exports_all_six_and_marks_misses(corpus,tmp_path,monkeypatch):
    checkpoint,calls=fake_pipeline(monkeypatch,tmp_path)
    output=tmp_path/'output'
    files=mapset.generate_mapset(checkpoint,corpus.parent/'0.wav',output,progress=lambda _:None)
    assert calls['loads']==calls['decode']==calls['mel']==calls['timing']==1
    assert len(calls['candidates'])==18
    assert len({id(c[3]) for c in calls['candidates']})==1
    assert len(files)==1 and len(list(output.glob('*.osz')))==1
    with zipfile.ZipFile(files[0]) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={'audio.ogg','generation.json','Easy.osu','Normal.osu','Hard.osu','Insane.osu','Expert.osu','ExpertPlus.osu'}
        metadata=json.loads(z.read('generation.json'))
        assert len(metadata['difficulties'])==6
        assert metadata['difficulties'][0]['seed']==44
        assert metadata['difficulties'][2]['in_band'] is False
        for tier in mapset.TIERS:
            bm=mapio.parse(z.read(tier.name.replace('+','Plus')+'.osu').decode())
            assert bm.general['AudioFilename']=='audio.ogg'
            assert bm.metadata['Creator']=='Lyra'
            if tier.name=='Hard':
                assert 'OUT OF RANGE' in bm.metadata['Version']
    assert load_json(output/'mapset-report.json')['status']=='complete'


def test_no_invalid_or_incomplete_mapset_is_exported(corpus,tmp_path,monkeypatch):
    checkpoint,_=fake_pipeline(monkeypatch,tmp_path,failures=(2.3,))
    output=tmp_path/'failed'
    with pytest.raises(ValueError,match='No complete mapset'):
        mapset.generate_mapset(checkpoint,corpus.parent/'0.wav',output,progress=lambda _:None)
    report=load_json(output/'mapset-report.json')
    assert report['status']=='failed' and not report['outputs']
    assert report['files'][0]['difficulties'][0]['name']=='Easy'
    assert list(output.glob('mapset-details-*/Easy.osu'))
    assert not list(output.glob('*.osz'))


def test_cancel_preserves_selected_difficulties_without_final_archive(corpus,tmp_path,monkeypatch):
    checkpoint,_=fake_pipeline(monkeypatch,tmp_path)
    output=tmp_path/'cancelled'
    stopped=False
    def progress(message):
        nonlocal stopped
        if message.startswith('Easy:'):
            stopped=True
    with pytest.raises(InterruptedError):
        mapset.generate_mapset(checkpoint,corpus.parent/'0.wav',output,progress=progress,cancelled=lambda:stopped)
    report=load_json(output/'mapset-report.json')
    assert report['status']=='cancelled'
    assert report['files'][0]['difficulties'][0]['name']=='Easy'
    assert not list(output.glob('*.osz')) and not list(output.glob('*.tmp'))


def test_api_modes_validation_persistence_and_default_single(tmp_path,monkeypatch):
    launched=[]
    def launch(home,action,args):
        launched.append((action,args))
        return 'fixture-job.json'
    monkeypatch.setattr(jobs,'start',launch)
    with TestClient(create_app(tmp_path)) as client:
        body=dict(checkpoint='weights.pt',inputs='music.mp3')
        assert client.post('/api/generate',json=body).status_code==200
        assert launched[-1][0]=='generate'
        assert client.post('/api/generate',json={**body,'mode':'mapset','targets':{'Expert+':8}}).status_code==200
        assert launched[-1][0]=='generate_mapset'
        assert launched[-1][1]['targets']['Expert+']==8
        assert load_json(tmp_path/'settings.json')['mode']=='mapset'
        assert client.get('/api/state').json()['settings']['targets']['Expert+']==8
        for bad in ({'mode':'bad'},{'mode':'mapset','targets':{'Easy':2}}):
            assert client.post('/api/generate',json={**body,**bad}).status_code==400
    assert len(launched)==2


def test_checkpoint_failure_is_saved_in_generation_report(tmp_path,monkeypatch):
    def missing(_):
        raise FileNotFoundError('missing checkpoint')
    monkeypatch.setattr(mapset,'load_checkpoint',missing)
    with pytest.raises(FileNotFoundError):
        mapset.generate_mapset('missing.pt','music.mp3',tmp_path)
    assert load_json(tmp_path/'mapset-report.json')['status']=='failed'

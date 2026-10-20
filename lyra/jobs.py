"""Detached durable workers; training has no dependency on a browser session."""
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid
import psutil
from filelock import FileLock
from .io import atomic_json, load_json, telemetry


def alive(state):
    try:
        process = psutil.Process(int(state['pid']))
        return process.is_running() and abs(process.create_time() - state['created']) < 1
    except (psutil.Error, KeyError, ValueError, TypeError):
        return False


def status(path):
    path = Path(path)
    spec = telemetry(path, {})
    state = telemetry(path.with_suffix('.status.json'), {})
    lease = state if state.get('pid') else spec
    if state.get('status') not in ('complete', 'cancelled', 'failed'):
        if alive(lease):
            state = {**state, 'status': 'running', 'pid': lease['pid'], 'created': lease['created']}
        elif spec.get('status') == 'starting' and time.time() - spec.get('started', 0) < 30:
            state = {**state, 'status': 'starting'}
        else:
            state = {**state, 'status': 'interrupted', 'message': 'A worker leállt; az utolsó mentésből folytatható.'}
    state.update(job_id=path.stem, action=spec.get('action'), arguments=spec.get('arguments', {}))
    log = path.with_suffix('.log')
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0, log.stat().st_size - 12000))
            state['log'] = f.read().decode('utf-8', errors='replace')
    return state


def latest(home):
    paths = sorted((Path(home) / 'jobs').glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    paths = [p for p in paths if not p.name.endswith('.status.json')]
    for path in paths:
        if status(path)['status'] in ('starting', 'running'):
            return path
    return paths[0] if paths else None


def start(home, action, arguments):
    root = Path(home).resolve() / 'jobs'
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / 'launch.lock'), timeout=5):
        current = latest(home)
        if current and status(current)['status'] in ('starting', 'running'):
            raise ValueError('Már fut egy Lyra-feladat. Várd meg vagy állítsd le a mentésével.')
        path = root / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6] + '.json')
        spec = dict(action=action, arguments=arguments, status='starting', started=time.time())
        atomic_json(path, spec)
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1',
               'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
        with path.with_suffix('.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-u', '-m', 'lyra', 'worker', str(path)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    cwd=Path(__file__).resolve().parents[1], env=env,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0),
                    start_new_session=os.name != 'nt')
        spec.update(pid=process.pid, created=psutil.Process(process.pid).create_time(), status='running')
        atomic_json(path, spec)
    return str(path)


def cancel(home):
    path = latest(home)
    if not path or status(path)['status'] not in ('running', 'starting'):
        return 'Nincs futó feladat.'
    path.with_suffix('.cancel').touch()
    return 'Leállítás kérve. A tanítás a lezárt update-eket menti.'


def dispatch(action, args, progress=print, cancelled=lambda: False):
    args = dict(args)
    if action == 'prepare':
        from .data import prepare
        return prepare(**args, progress=progress, cancelled=cancelled)
    if action == 'train':
        from .training import train
        from .config import TrainConfig
        args['cfg'] = TrainConfig(**args.pop('config', {}))
        return train(**args, progress=progress, cancelled=cancelled)
    if action == 'generate':
        from .generation import generate, GenerationConfig
        args['cfg'] = GenerationConfig(**args.pop('config', {}))
        return generate(**args, progress=progress, cancelled=cancelled)
    if action == 'benchmark':
        from .benchmark import benchmark
        return benchmark(**args, progress=progress, cancelled=cancelled)
    if action == 'evaluate':
        from .evaluation import evaluate
        return evaluate(**args, progress=progress, cancelled=cancelled)
    raise ValueError('Ismeretlen Lyra-feladat.')


def worker(path):
    path = Path(path)
    spec = load_json(path)
    lease = dict(pid=os.getpid(), created=psutil.Process().create_time())
    def progress(message):
        print(message, flush=True)
        atomic_json(path.with_suffix('.status.json'), dict(status='running', message=str(message), **lease))
    try:
        progress('Lyra háttérfolyamat elindult.')
        result = dispatch(spec['action'], spec['arguments'], progress, lambda: path.with_suffix('.cancel').exists())
        atomic_json(path.with_suffix('.status.json'), dict(status='cancelled' if path.with_suffix('.cancel').exists() else 'complete', result=result, **lease))
    except (InterruptedError, KeyboardInterrupt) as exc:
        atomic_json(path.with_suffix('.status.json'), dict(status='cancelled', message=str(exc), **lease))
    except Exception as exc:
        traceback.print_exc()
        atomic_json(path.with_suffix('.status.json'), dict(status='failed', error=str(exc), **lease))
        raise

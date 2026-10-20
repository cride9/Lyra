"""Shared device lease and resumable single-model checkpoints."""
from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
import random
import tempfile
import time
import numpy as np
import psutil
import torch
from filelock import FileLock
from .tokenizer import VERSION


def device_for(name='auto'):
    name = ('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else name
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA nem érhető el ebben a Python-környezetben.')
    return torch.device(name)


def amp(device):
    return torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()


@contextmanager
def gpu_session(device, fraction=.8):
    if device.type != 'cuda':
        yield
        return
    # Honour both the legacy worker lease and its durable jobs, including jobs
    # whose launcher PID differs from the actual Python interpreter on Windows.
    legacy = Path(tempfile.gettempdir()) / 'osu-mapper-v2-gpu.lock'
    if legacy.exists():
        try:
            pid = int(legacy.read_text())
        except (OSError, ValueError):
            pid = -1
        if psutil.pid_exists(pid):
            raise ValueError('Még fut a korábbi mapper a GPU-n. Előbb állítsd le a mentésével.')
    lock = FileLock(str(Path(tempfile.gettempdir()) / 'lyra-gpu.lock'), timeout=0)
    with lock:
        free, total = torch.cuda.mem_get_info(device)
        if free < total * .65:
            raise ValueError('A GPU memóriáját más program használja. A próba/tanítás később indítható.')
        torch.cuda.set_per_process_memory_fraction(fraction, device.index or 0)
        yield


def move(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def device_batches(batches, device):
    if device.type != 'cuda':
        yield from (move(b, device) for b in batches)
        return
    stream = torch.cuda.Stream(device=device)
    iterator = iter(batches)
    def preload():
        cpu = next(iterator, None)
        if cpu is None:
            return None
        with torch.cuda.stream(stream):
            return move(cpu, device)
    pending = preload()
    while pending is not None:
        current = torch.cuda.current_stream(device)
        current.wait_stream(stream)
        batch = pending
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                value.record_stream(current)
        pending = preload()
        yield batch


def random_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_random(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    try:
        with tmp.open('wb') as f:
            torch.save(state, f); f.flush(); os.fsync(f.fileno())
        for attempt in range(8):
            try:
                tmp.replace(path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(.05 * (attempt + 1))
    finally:
        tmp.unlink(missing_ok=True)


def load_checkpoint(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('format') != 'lyra-1' or state.get('tokenizer') != VERSION:
        raise ValueError('Ez nem Lyra-mentés. A V1/V2 súlyai nem tölthetők be más architektúrába.')
    return state


def memory(device):
    return dict(peak_vram_gib=torch.cuda.max_memory_allocated(device) / 1024 ** 3,
                reserved_vram_gib=torch.cuda.memory_reserved(device) / 1024 ** 3,
                free_vram_gib=torch.cuda.mem_get_info(device)[0] / 1024 ** 3) if device.type == 'cuda' else {}

from __future__ import annotations

import hashlib
import math
import os
import subprocess
from functools import lru_cache

import numpy as np

from .mapio import Beatmap

SR, HOP, FFT, MELS = 22050, 220, 1024, 128
FRAME_MS = HOP * 1000 / SR
VERSION = "mel-v1-22050-220-1024-128-centered"
EXTENSIONS = {".mp3", ".ogg", ".wav", ".flac"}


def ffmpeg():
    import imageio_ffmpeg
    return os.environ.get("OSUMAPPER_FFMPEG") or imageio_ffmpeg.get_ffmpeg_exe()


def decode(path, sr=SR):
    args = [ffmpeg(), "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    result = subprocess.run(args, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=600)
    if result.returncode:
        raise ValueError(f"Audio decode failed: {result.stderr.decode(errors='replace')[-500:]}")
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Empty or non-finite audio")
    return audio


@lru_cache(maxsize=1)
def filterbank():
    hz_to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    edges = 700 * (10 ** (np.linspace(hz_to_mel(20), hz_to_mel(SR / 2), MELS + 2) / 2595) - 1)
    freq = np.fft.rfftfreq(FFT, 1 / SR)
    filters = np.zeros((MELS, len(freq)), np.float32)
    for i in range(MELS):
        filters[i] = np.maximum(0, np.minimum((freq - edges[i]) / (edges[i + 1] - edges[i]), (edges[i + 2] - freq) / (edges[i + 2] - edges[i + 1])))
        filters[i] *= 2 / (edges[i + 2] - edges[i])
    filters.setflags(write=False)
    return filters


@lru_cache(maxsize=1)
def fft_window():
    value = np.hanning(FFT).astype(np.float32)
    value.setflags(write=False)
    return value


def spectrogram_block(pcm, start, count=4096):
    """Global frame indices, with exactly the full recording's centered padding."""
    frames_total = len(pcm) // HOP + 1
    count = min(count, frames_total - start)
    if start < 0 or count <= 0:
        raise ValueError('Spectrogram block lies outside the recording')
    left = start * HOP - FFT // 2
    right = (start + count - 1) * HOP + FFT // 2
    segment = pcm[max(0, left):min(len(pcm), right)]
    segment = np.pad(segment, (max(0, -left), max(0, right - len(pcm))))
    frames = np.lib.stride_tricks.sliding_window_view(segment, FFT)[::HOP]
    power = np.abs(np.fft.rfft(frames * fft_window(), axis=1)) ** 2
    return np.log(np.maximum(power @ filterbank().T, 1e-8)).astype(np.float32).T


def spectrogram(audio):
    # Frame i is centered at i*HOP samples. Do not trim or normalize silence.
    return np.concatenate([spectrogram_block(audio, start)
                           for start in range(0, len(audio) // HOP + 1, 4096)], axis=1)


def pcm_hash(audio):
    return hashlib.sha256(np.asarray(np.clip(audio, -1, 1) * 32767, dtype="<i2").tobytes()).hexdigest()


def fingerprint(audio):
    """Cheap duplicate-recording fingerprint; metadata also groups song edits."""
    count = len(audio) // SR
    if count < 4:
        return pcm_hash(audio)
    blocks = audio[:count * SR].reshape(count, SR)
    energy = np.log(np.maximum((blocks ** 2).mean(1), 1e-8))
    energy = (energy - energy.mean()) / max(float(energy.std()), 0.01)
    return hashlib.sha256(np.round(energy * 2).astype(np.int8).tobytes()).hexdigest()


def window(mel, base_ms, duration_ms=64000):
    n = math.ceil(duration_ms / FRAME_MS / 4) * 4
    start = round(base_ms / FRAME_MS)
    result = np.full((MELS, n), np.log(1e-8), dtype=np.float32)
    left, right = max(0, start), min(mel.shape[1], start + n)
    if right > left:
        result[:, left - start:right - start] = mel[:, left:right]
    # Fixed scaling avoids a train/inference mismatch and keeps padding quiet.
    return (result + 5) / 7


def invalid_beat_mask(bm: Beatmap, frames: int):
    """Do not supervise unresolvable (<1 ms) or invalid timing segments."""
    times=np.arange(frames)*FRAME_MS
    invalid=np.zeros(frames,dtype=bool)
    reds=[p for p in bm.timing if p.uninherited]
    for i,p in enumerate(reds):
        if np.isfinite(p.beat_length) and p.beat_length>=1: continue
        end=reds[i+1].time if i+1<len(reds) else float('inf')
        invalid |= (times>=p.time-4*FRAME_MS)&(times<=end+4*FRAME_MS)
    return invalid


def beat_targets(bm: Beatmap, frames: int):
    times = np.arange(frames) * FRAME_MS
    targets = np.zeros((frames, 2), np.float32)
    if not frames: return targets
    reds = [p for p in bm.timing if p.uninherited]
    for i, red in enumerate(reds):
        if not np.isfinite(red.beat_length) or red.beat_length<1: continue
        end = reds[i + 1].time if i + 1 < len(reds) else times[-1] + FRAME_MS
        # Beats farther away cannot reach any frame through the 7-frame kernel.
        end = min(end,times[-1]+4*FRAME_MS)
        first = int(np.floor((0 - red.time) / red.beat_length)) if red.time < 0 else 0
        for beat_idx in range(first, int(np.ceil((end - red.time) / red.beat_length))):
            at = red.time + beat_idx * red.beat_length
            if at < 0 or at >= end:
                continue
            center = round(at / FRAME_MS)
            for j in range(max(0, center - 3), min(frames, center + 4)):
                value = np.exp(-0.5 * ((times[j] - at) / 15) ** 2)
                targets[j, 0] = max(targets[j, 0], value)
                if beat_idx % red.meter == 0:
                    targets[j, 1] = max(targets[j, 1], value)
    return targets


def phase_features(bm: Beatmap, base_ms, frames):
    times = base_ms + np.arange(frames) * FRAME_MS
    result = np.zeros((frames, 3), np.float32)
    reds = [p for p in bm.timing if p.uninherited]
    for i, red in enumerate(reds):
        end = reds[i + 1].time if i + 1 < len(reds) else np.inf
        mask = (times >= (red.time if i else -np.inf)) & (times < end)
        phase = (times[mask] - red.time) / red.beat_length
        result[mask, 0] = np.sin(phase * 2 * np.pi)
        result[mask, 1] = np.cos(phase * 2 * np.pi)
        result[mask, 2] = np.cos(phase * 2 * np.pi / max(1, red.meter))
    return result


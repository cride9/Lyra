"""One context builder for teacher forcing and free generation; past only."""
import math
import numpy as np
from . import audio
from .style import describe
from .tokenizer import Tokenizer


def summaries(bm, base, start, limit=128):
    stop = max(0, base)
    if stop <= 0:
        return np.zeros((1, 68), np.float32), np.zeros(1, bool)
    stride = max(8000, math.ceil(stop / (limit * 8000)) * 8000)
    rows = []
    for left in range(0, stop, stride):
        right = min(left + stride, stop)
        at = (left + right) / 2
        age = (at - start) / 64000
        rows.append(np.r_[describe(bm, left, right), age, np.sin(age), np.cos(age),
                          min(1, (right - left) / 64000)])
    return np.asarray(rows, np.float32), np.ones(len(rows), bool)


def conditioning(bm, start, end, duration, stars, styles, settings, config):
    tok = Tokenizer()
    base = max(0, int(start) - config.history_ms)
    # The same complete objects, active sliders, and summary cutoffs are used
    # during training and generation. No parallel cursor-state convention.
    history = tok.context(bm, base, start, config.history_tokens)
    prefix = tok.condition(stars, styles, settings) + tok.time(end - 1, base)
    prefix += [tok.ids['CTX']] + history + [tok.ids['GEN']]
    past, past_valid = summaries(bm, base, start, config.summary_tokens)
    return dict(base=base, prefix=prefix, summary=past, summary_valid=past_valid,
                start=int(start), end=int(end), duration=float(duration))


def windows(bm, duration, config):
    tok = Tokenizer()
    events = [(o.time, len(tok.object(bm, o, o.time))) for o in bm.objects]
    events += [(a, 7) for a, _ in bm.breaks]
    sizes = {}
    for at, size in events:
        sizes[at] = sizes.get(at, 0) + size
    times = sorted(sizes)
    budget = config.max_tokens - config.history_tokens - 15
    if any(n > budget for n in sizes.values()):
        raise ValueError('Simultaneous objects exceed the Lyra token budget.')
    start, cursor = 0, 0
    result = []
    while start < duration:
        end = min(int(math.ceil(duration)), start + config.output_ms)
        used, at = 0, cursor
        while at < len(times) and times[at] < end:
            if times[at] >= start:
                if used + sizes[times[at]] > budget:
                    end = times[at]
                    break
                used += sizes[times[at]]
            at += 1
        if end <= start:
            raise ValueError('The map window is not advancing.')
        result.append((start, end, used + config.history_tokens + 14))
        start, cursor = end, at
    return result


def phase(bm, base, count):
    return audio.phase_features(bm, base, count)

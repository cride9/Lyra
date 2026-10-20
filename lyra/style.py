"""Explicit 64-value, tempo/CS-normalized pattern descriptor."""
from __future__ import annotations

import numpy as np
from .geometry import radius


def _hist(values, edges):
    counts = np.histogram(values, edges)[0].astype(np.float32)
    return counts / max(1, counts.sum())


def describe(bm, start=-float("inf"), end=float("inf"), exclude=None):
    objects = [o for o in bm.objects if start <= o.time < end and not (exclude and exclude[0] <= o.time < exclude[1])]
    if not objects:
        return np.zeros(64, np.float32)
    times = np.array([o.time for o in objects])
    xy = np.array([(o.x, o.y) for o in objects])
    beats = np.array([bm.clock_at(t)[0] for t in times])
    delta = np.diff(xy, axis=0)
    distance = np.linalg.norm(delta, axis=1) / (2 * radius(bm.difficulty.get("CircleSize", 4)))
    angle = np.arctan2(delta[:, 1], delta[:, 0]) if len(delta) else []
    turns = (np.diff(angle) + np.pi) % (2*np.pi) - np.pi
    sliders = [o for o in objects if o.kind == "slider"]
    # 12 + 10 + 12 + 4 + 4 + 8 + 3 + 1 + 9 + 1 = 64.
    grid = np.histogram2d(xy[:, 0], xy[:, 1], bins=(np.linspace(0, 512, 4), np.linspace(0, 384, 4)))[0].ravel()
    parts = [
        _hist(np.diff(times) / beats[:-1], [0,.125,.167,.25,.333,.5,.667,.75,1,1.5,2,4,np.inf]),
        _hist(distance, [0,.25,.5,1,1.5,2,3,4,6,8,np.inf]),
        _hist(turns, np.linspace(-np.pi, np.pi, 13)),
        _hist(["BCLP".index(o.curve) for o in sliders], np.arange(5)-.5),
        _hist([o.repeats for o in sliders], [.5,1.5,2.5,4.5,np.inf]),
        _hist([o.length / (2*radius(bm.difficulty.get("CircleSize", 4))) for o in sliders], [0,.5,1,1.5,2,3,4,6,np.inf]),
        np.array([sum(o.kind == k for o in objects)/len(objects) for k in ("circle","slider","spinner")]),
        [sum(o.new_combo for o in objects)/len(objects)], grid/max(1, grid.sum()),
        [min(1, len(objects) / max(1, (times[-1]-times[0])/np.median(beats)) / 16)],
    ]
    return np.concatenate(parts).astype(np.float32)


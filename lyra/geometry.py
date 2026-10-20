"""Flatten native osu! paths, including expected-distance trimming/extension."""
from __future__ import annotations

import math
import numpy as np


def _bezier(points, tolerance=0.1, depth=0):
    p = np.asarray(points, dtype=np.float64)
    if len(p) <= 2:
        return list(p)
    chord = p[-1] - p[0]
    length = np.linalg.norm(chord)
    error = np.max(np.abs(chord[0] * (p[:, 1] - p[0, 1]) - chord[1] * (p[:, 0] - p[0, 0]))) / max(length, 1e-12)
    polygon = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    if depth >= 18 or (error <= tolerance and polygon - length <= tolerance):
        return [p[0], p[-1]]
    left, right, row = [p[0]], [p[-1]], p
    while len(row) > 1:
        row = (row[:-1] + row[1:]) * 0.5
        left.append(row[0]); right.append(row[-1])
    return _bezier(left, tolerance, depth + 1)[:-1] + _bezier(right[::-1], tolerance, depth + 1)


def _perfect(p):
    a, b, c = p
    matrix = 2 * np.stack([b - a, c - a])
    if abs(np.linalg.det(matrix)) < 1e-7:
        return _bezier(p)
    center = np.linalg.solve(matrix, np.array([b @ b - a @ a, c @ c - a @ a]))
    radius = np.linalg.norm(a - center)
    angles = np.arctan2(p[:, 1] - center[1], p[:, 0] - center[0])
    sweep = (angles[2] - angles[0]) % (2 * math.pi)
    if (angles[1] - angles[0]) % (2 * math.pi) > sweep:
        sweep -= 2 * math.pi
    step = 2 * math.acos(max(-1, min(1, 1 - 0.05 / max(radius, 0.05))))
    count = min(100000, max(2, math.ceil(abs(sweep) / max(step, 1e-4)) + 1))
    theta = np.linspace(angles[0], angles[0] + sweep, count)
    return center + radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)


def raw_path(obj):
    p = np.array([(obj.x, obj.y), *obj.points], dtype=np.float64)
    if obj.curve == "L":
        return p
    if obj.curve == "P" and len(p) == 3:
        return np.asarray(_perfect(p))
    if obj.curve == "C":
        result = []
        for i in range(len(p) - 1):
            a, b, c = p[max(0, i - 1)], p[i], p[i + 1]
            d = p[i + 2] if i + 2 < len(p) else c + (c - b)
            for t in np.linspace(0, 1, 101)[:-1]:
                result.append(0.5 * ((2*b) + (-a+c)*t + (2*a-5*b+4*c-d)*t*t + (-a+3*b-3*c+d)*t*t*t))
        return np.asarray([*result, p[-1]])
    result, first = [], 0
    for i in range(1, len(p)):
        if i == len(p) - 1 or np.array_equal(p[i], p[i + 1]):
            result.extend(_bezier(p[first:i + 1]))
            first = i + 1
    return np.asarray(result)


def slider_path(obj):
    if obj.kind != "slider" or obj.length <= 0:
        raise ValueError("Expected a positive-length slider")
    points = raw_path(obj)
    if not np.isfinite(points).all() or len(points) < 2:
        raise ValueError("Degenerate slider path")
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.r_[True, distances > 1e-9]
    points = points[keep]
    if len(points) < 2:
        raise ValueError("Zero-length slider geometry")
    cumulative = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if obj.length < cumulative[-1]:
        i = np.searchsorted(cumulative, obj.length, side="right")
        fraction = (obj.length - cumulative[i - 1]) / (cumulative[i] - cumulative[i - 1])
        return np.vstack([points[:i], points[i - 1] + fraction * (points[i] - points[i - 1])])
    if obj.length > cumulative[-1]:
        # osu! cannot extend past an explicitly repeated final anchor.
        anchors = [(obj.x, obj.y), *obj.points]
        if len(anchors) > 1 and anchors[-1] == anchors[-2]:
            return points
        direction = (points[-1] - points[-2]) / np.linalg.norm(points[-1] - points[-2])
        points = np.vstack([points, points[-1] + direction * (obj.length - cumulative[-1])])
    return points


def radius(cs):
    return max(1.0, 54.4 - 4.48 * float(cs))


def validate_object(obj, cs=4):
    if obj.kind == "spinner":
        return
    points = slider_path(obj) if obj.kind == "slider" else np.array([[obj.x, obj.y]])
    r = radius(cs) + (0.25 if obj.kind == "slider" else 0)
    if not np.isfinite(points).all() or np.any(points < [r, r]) or np.any(points > [512-r, 384-r]):
        raise ValueError(f"Out-of-bounds {obj.kind} at {obj.time} ms (played path + CS radius)")


def validate_map(bm, duration_ms=None):
    bm.validate(duration_ms, generated=True)
    for obj in bm.objects:
        validate_object(obj, bm.difficulty.get("CircleSize", 4))
    for a, b in bm.breaks:
        if duration_ms is not None and b > duration_ms:
            raise ValueError("Break extends beyond audio")

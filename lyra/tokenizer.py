"""Typed events, exact millisecond times, and a finite-state generation grammar.

Slider anchors retain their native curve type and repeated Bezier separators.
Coordinates are rounded to the nearest 2 osu! pixels; custom samples are omitted.
"""
from __future__ import annotations

from dataclasses import replace
import math
from .geometry import radius
from .mapio import HitObject, TimingPoint

VERSION = "lyra-events-v1-64s-ms-xy2"
STYLES = ("aim", "streams", "rhythm")


class Tokenizer:
    def __init__(self):
        self.names = ["PAD", "BOS", "CTX", "GEN", "EOS", "END", "PATH_END", "CIRCLE", "SLIDER", "SPINNER", "BREAK"]
        self.ranges = {}
        for name, count in [("T", 6401), ("F", 10), ("XY", 2049), ("COMBO", 16), ("HS", 16), ("CURVE", 4), ("REP", 64), ("BYTE", 256), ("STAR", 202), ("AIM", 4), ("STREAMS", 4), ("RHYTHM", 4), ("AR", 112), ("OD", 112), ("CS", 112), ("HP", 112)]:
            self.ranges[name] = (len(self.names), count)
            self.names += [f"{name}_{i}" for i in range(count)]
        self.ids = {n: i for i, n in enumerate(self.names)}

    def __len__(self):
        return len(self.names)

    def t(self, group, value):
        first, size = self.ranges[group]
        if not 0 <= value < size:
            raise ValueError(f"{group} value {value} outside vocabulary")
        return first + int(value)

    def v(self, group, token):
        value = token - self.ranges[group][0]
        if not 0 <= value < self.ranges[group][1]:
            raise ValueError(f"Expected {group}, got {self.names[token]}")
        return value

    def group(self, name, low=0, high=None):
        start, size = self.ranges[name]
        return list(range(start + max(0, low), start + min(size - 1, size - 1 if high is None else high) + 1))

    def integer(self, value):
        value = round(value)
        if not 0 <= value < 2**24:
            raise ValueError("Numeric event exceeds 24-bit range")
        return [self.t("BYTE", (value >> shift) & 255) for shift in (16, 8, 0)]

    def read_integer(self, values):
        if len(values) != 3:
            raise ValueError("Truncated numeric token")
        return sum(self.v("BYTE", t) << shift for t, shift in zip(values, (16, 8, 0)))

    def condition(self, stars=None, styles=None, settings=None):
        styles, settings = styles or {}, settings or {}
        out = [self.ids["BOS"], self.t("STAR", 201 if stars is None else min(200, max(0, round(stars * 10))))]
        for key in STYLES:
            value = styles.get(key)
            out.append(self.t(key.upper(), 0 if value is None else int(value) + 1))
        for key in ("AR", "OD", "CS", "HP"):
            value = settings.get(key)
            out.append(self.t(key, 111 if value is None else min(110, max(0, round(value * 10)))))
        return out

    def time(self, ms, base):
        relative = round(ms - base)
        if not 0 <= relative <= 64000:
            raise ValueError("Event outside audio window")
        return [self.t("T", relative // 10), self.t("F", relative % 10)]

    def xy(self, x, y):
        return [self.t("XY", round(x / 2) + 1024), self.t("XY", round(y / 2) + 1024)]

    def object(self, bm, obj, base):
        out = [self.ids[obj.kind.upper()]] + self.time(obj.time, base)
        out += self.xy(obj.x, obj.y) + [self.t("COMBO", int(obj.new_combo) * 8 + obj.combo_skip), self.t("HS", obj.hitsound & 15)]
        if obj.kind == "slider":
            out += [self.t("CURVE", "BCLP".index(obj.curve)), self.t("REP", obj.repeats - 1)]
            out += self.integer(max(1, bm.duration(obj))) + self.integer(bm.clock_at(obj.time)[1] * 10000)
            for x, y in obj.points:
                out += self.xy(x, y)
            out += [self.ids["PATH_END"]]
        elif obj.kind == "spinner":
            out += self.integer(obj.end_time - obj.time)
        return out + [self.ids["END"]]

    def records(self, bm, base, start, end):
        records = [(o.time, self.object(bm, o, base)) for o in bm.objects if start <= o.time < end]
        records += [(a, [self.ids["BREAK"]] + self.time(a, base) + self.integer(b - a) + [self.ids["END"]]) for a, b in bm.breaks if start <= a < end]
        return sorted(records, key=lambda x: x[0])

    def context(self, bm, base, start, limit=2048):
        records = self.records(bm, base, base, start)
        # An active object may have started before the audio window. Encode its
        # remaining duration at the window origin, rather than forgetting it.
        for obj in bm.objects:
            finish = obj.time + bm.duration(obj)
            if obj.time < base and finish > start:
                active = replace(obj, time=round(base), end_time=round(finish))
                if obj.kind == "slider":
                    beat, sv = bm.clock_at(base)
                    active.length = (finish - base) / active.repeats * float(bm.difficulty["SliderMultiplier"]) * 100 * sv / beat
                records.insert(0, (base, self.object(bm, active, base)))
        selected = []
        size = 0
        for _, record in reversed(records):
            if size + len(record) > limit:
                break
            selected.insert(0, record)
            size += len(record)
        return [v for r in selected for v in r]

    def decode(self, tokens, bm, base):
        """Decode only complete records; reject malformed/truncated sequences."""
        i = 0
        while i < len(tokens):
            kind = self.names[tokens[i]]
            i += 1
            if kind == "EOS":
                if i != len(tokens):
                    raise ValueError("Data after EOS")
                break
            if kind not in ("CIRCLE", "SLIDER", "SPINNER", "BREAK"):
                raise ValueError(f"Unexpected event {kind}")
            time = round(base + self.v("T", tokens[i]) * 10 + self.v("F", tokens[i + 1]))
            i += 2
            if kind == "BREAK":
                duration = self.read_integer(tokens[i:i + 3]); i += 3
                bm.breaks.append((time, time + duration))
            else:
                x, y = [(self.v("XY", t) - 1024) * 2 for t in tokens[i:i + 2]]; i += 2
                combo, hs = self.v("COMBO", tokens[i]), self.v("HS", tokens[i + 1]); i += 2
                obj = HitObject(x, y, time, kind.lower(), bool(combo // 8), combo % 8, hs)
                if kind == "SLIDER":
                    obj.curve = "BCLP"[self.v("CURVE", tokens[i])]
                    obj.repeats = self.v("REP", tokens[i + 1]) + 1; i += 2
                    duration = self.read_integer(tokens[i:i + 3]); i += 3
                    sv = self.read_integer(tokens[i:i + 3]) / 10000; i += 3
                    if not 0.1 <= sv <= 10 or duration <= 0:
                        raise ValueError("Invalid slider duration or SV")
                    while tokens[i] != self.ids["PATH_END"]:
                        obj.points.append(tuple((self.v("XY", t) - 1024) * 2 for t in tokens[i:i + 2])); i += 2
                    i += 1
                    beat = bm.clock_at(time)[0]
                    obj.length = duration / obj.repeats * float(bm.difficulty["SliderMultiplier"]) * 100 * sv / beat
                    bm.timing.append(TimingPoint(time, -100 / sv, uninherited=False))
                    bm.timing.sort(key=lambda p: p.time)
                elif kind == "SPINNER":
                    obj.end_time = time + self.read_integer(tokens[i:i + 3]); i += 3
                bm.objects.append(obj)
            if i >= len(tokens) or tokens[i] != self.ids["END"]:
                raise ValueError("Unterminated object")
            i += 1
        bm.objects.sort(key=lambda o: o.time)
        return bm


class Grammar:
    """Incremental token mask; no text parsing or silent event repair."""
    def __init__(self, tok, base, start, end, audio_end, busy_until=0, *, circle_size=None):
        self.tok, self.base, self.end = tok, base, end
        self.audio_end = audio_end
        self.circle_size = circle_size
        self.minimum = max(start, busy_until)
        self.state, self.kind = "event", ""
        self.coarse, self.at = 0, start
        self.prefix, self.count, self.anchors = 0, 0, 0

    def allowed(self, remaining=1536):
        t, state = self.tok, self.state
        if state == "done":
            return []
        if state == "event":
            # Reserve enough room to complete the longest minimal event + EOS.
            return [t.ids["EOS"]] + ([] if self.minimum >= self.end or remaining < 32 else [t.ids[k] for k in ("CIRCLE", "SLIDER", "SPINNER", "BREAK")])
        if state == "time":
            return t.group("T", int(max(0, self.minimum - self.base)) // 10, int(self.end - 1 - self.base) // 10)
        if state == "fine":
            return [t.t("F", n) for n in range(10) if self.minimum <= self.base + self.coarse * 10 + n < self.end]
        if state in ("x", "y", "anchor_x", "anchor_y"):
            if state.startswith("anchor"):
                allowed = t.group("XY", 768, 1536)  # anchors may be outside playfield
                if state == "anchor_x" and self.anchors:
                    if remaining < 5 or self.anchors >= 64:
                        return [t.ids["PATH_END"]]
                    allowed += [t.ids["PATH_END"]]
                return allowed
            extent = 512 if state == "x" else 384
            margin = 0
            if self.circle_size is not None and self.kind in ("CIRCLE", "SLIDER"):
                margin = radius(self.circle_size) + (0.25 if self.kind == "SLIDER" else 0)
            return t.group("XY", 1024 + math.ceil(margin / 2),
                           1024 + math.floor((extent - margin) / 2))
        if state in ("combo", "hs", "curve", "rep"):
            return t.group({"combo": "COMBO", "hs": "HS", "curve": "CURVE", "rep": "REP"}[state])
        if state in ("duration", "sv"):
            lo, hi = (1, max(1, min(2**24 - 1, int(self.audio_end - self.at)))) if state == "duration" else (1000, 100000)
            shift = (2 - self.count) * 8
            return [t.t("BYTE", n) for n in range(256) if ((self.prefix * 256 + n) << shift) <= hi and (((self.prefix * 256 + n + 1) << shift) - 1) >= lo]
        if state == "end":
            return [t.ids["END"]]
        raise ValueError(state)

    def consume(self, token):
        t, s = self.tok, self.state
        if s == "event":
            self.kind = t.names[token]
            self.state = "done" if self.kind == "EOS" else "time"
        elif s == "time":
            self.coarse = t.v("T", token); self.state = "fine"
        elif s == "fine":
            self.at = self.base + self.coarse * 10 + t.v("F", token)
            self.state = "duration" if self.kind == "BREAK" else "x"
        elif s in ("x", "y", "combo"):
            self.state = {"x": "y", "y": "combo", "combo": "hs"}[s]
        elif s == "hs":
            self.state = {"CIRCLE": "end", "SLIDER": "curve", "SPINNER": "duration"}[self.kind]
        elif s == "curve":
            self.state = "rep"
        elif s == "rep":
            self.state = "duration"
        elif s in ("duration", "sv"):
            self.prefix = self.prefix * 256 + t.v("BYTE", token); self.count += 1
            if self.count == 3:
                if s == "duration":
                    self.finish = self.at + self.prefix
                self.prefix, self.count = 0, 0
                self.state = ("sv" if self.kind == "SLIDER" else "end") if s == "duration" else "anchor_x"
        elif s == "anchor_x":
            self.state = "end" if token == t.ids["PATH_END"] else "anchor_y"
        elif s == "anchor_y":
            self.anchors += 1; self.state = "anchor_x"
        elif s == "end":
            # Generated notes do not overlap an ongoing slider or spinner.
            self.minimum = max(self.at + 1, getattr(self, "finish", 0) if self.kind != "CIRCLE" else 0)
            self.anchors = 0; self.state = "event"

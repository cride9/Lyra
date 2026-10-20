from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TimingPoint:
    time: float
    beat_length: float
    meter: int = 4
    sample_set: int = 1
    sample_index: int = 0
    volume: int = 70
    uninherited: bool = True
    effects: int = 0


@dataclass
class HitObject:
    x: float
    y: float
    time: int
    kind: str = "circle"
    new_combo: bool = False
    combo_skip: int = 0
    hitsound: int = 0
    curve: str = "B"
    points: list[tuple[float, float]] = field(default_factory=list)
    repeats: int = 1
    length: float = 0
    end_time: int = 0
    edge_sounds: list[int] = field(default_factory=list)
    edge_sets: list[str] = field(default_factory=list)
    sample: str = "0:0:0:0:"


@dataclass
class Beatmap:
    general: dict = field(default_factory=lambda: {"AudioFilename": "audio.ogg", "Mode": "0", "StackLeniency": "0.7"})
    metadata: dict = field(default_factory=lambda: {"Title": "Untitled", "Artist": "Unknown", "Creator": "Lyra", "Version": "Generated"})
    difficulty: dict = field(default_factory=lambda: {"HPDrainRate": "5", "CircleSize": "4", "OverallDifficulty": "7", "ApproachRate": "8", "SliderMultiplier": "1.4", "SliderTickRate": "1"})
    timing: list[TimingPoint] = field(default_factory=list)
    objects: list[HitObject] = field(default_factory=list)
    breaks: list[tuple[int, int]] = field(default_factory=list)
    source: str = ""

    def clock_at(self, time: float) -> tuple[float, float]:
        """Red points reset inherited SV; preserve same-time file ordering."""
        red = next((p for p in self.timing if p.uninherited), None)
        if red is None:
            raise ValueError("No uninherited timing point")
        beat, sv = red.beat_length, 1.0
        for p in sorted(self.timing, key=lambda p: p.time):
            if p.time > time:
                break
            if p.uninherited:
                beat, sv = p.beat_length, 1.0
            else:
                sv = min(10.0, max(0.1, -100.0 / p.beat_length))
        return beat, sv

    def duration(self, obj: HitObject) -> float:
        if obj.kind == "slider":
            beat, sv = self.clock_at(obj.time)
            return obj.length * obj.repeats * beat / (100 * float(self.difficulty.get("SliderMultiplier", 1.4)) * sv)
        return max(0, obj.end_time - obj.time) if obj.kind == "spinner" else 0.0

    def copy(self) -> Beatmap:
        return copy.deepcopy(self)

    def validate(self, audio_ms: float | None = None, generated: bool = False) -> None:
        if int(self.general.get("Mode", 0)) != 0:
            raise ValueError("Only osu!standard (Mode:0) is supported")
        if not any(p.uninherited for p in self.timing):
            raise ValueError("Missing timing points")
        for p in self.timing:
            if not all(math.isfinite(x) for x in [p.time, p.beat_length]) or p.beat_length == 0:
                raise ValueError("Non-finite or zero timing value")
            if p.uninherited != (p.beat_length > 0) or p.meter <= 0:
                raise ValueError("Invalid timing point")
        last = -math.inf
        for obj in self.objects:
            if obj.time < last or obj.time < 0:
                raise ValueError("Unsorted or negative hit object time")
            last = obj.time
            if not all(math.isfinite(v) for v in [obj.x, obj.y, obj.length]):
                raise ValueError("Non-finite hit object")
            if obj.kind == "slider" and (obj.curve not in "BCLP" or not obj.points or obj.repeats < 1 or obj.length <= 0):
                raise ValueError("Invalid slider")
            if any(not math.isfinite(x) or not math.isfinite(y) for x, y in obj.points):
                raise ValueError("Non-finite slider anchor")
            if obj.kind == "spinner" and obj.end_time <= obj.time:
                raise ValueError("Invalid spinner duration")
            if audio_ms is not None and obj.time + self.duration(obj) > audio_ms + 100:
                raise ValueError("Hit object extends beyond audio")
            if generated and not (0 <= obj.x <= 512 and 0 <= obj.y <= 384):
                raise ValueError("Generated object outside playfield")
        for start, end in self.breaks:
            if start < 0 or end <= start:
                raise ValueError("Invalid break")


def parse(text: str, source: str = "") -> Beatmap:
    bm = Beatmap(source=source)
    section = ""
    for line_number, raw in enumerate(text.lstrip("\ufeff").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        try:
            if section in ("General", "Metadata", "Difficulty") and ":" in line:
                key, value = line.split(":", 1)
                getattr(bm, section.lower())[key.strip()] = value.strip()
            elif section == "TimingPoints":
                f = line.split(",")
                defaults = ["0", "500", "4", "1", "0", "70", "1", "0"]
                f += defaults[len(f):]
                bm.timing.append(TimingPoint(float(f[0]), float(f[1]), int(f[2]), int(f[3]), int(f[4]), int(f[5]), f[6] == "1", int(f[7])))
            elif section == "Events" and line.split(",", 1)[0] in ("2", "Break"):
                f = line.split(",")
                bm.breaks.append((int(f[1]), int(f[2])))
            elif section == "HitObjects":
                f = line.split(",")
                flags = int(f[3])
                obj = HitObject(float(f[0]), float(f[1]), int(float(f[2])), new_combo=bool(flags & 4), combo_skip=(flags >> 4) & 7, hitsound=int(f[4]))
                if flags & 1:
                    obj.sample = f[5] if len(f) > 5 else obj.sample
                elif flags & 2:
                    obj.kind = "slider"
                    path = f[5].split("|")
                    obj.curve = path[0]
                    obj.points = [tuple(map(float, p.split(":"))) for p in path[1:]]
                    obj.repeats, obj.length = int(f[6]), float(f[7])
                    obj.edge_sounds = list(map(int, f[8].split("|"))) if len(f) > 8 and f[8] else []
                    obj.edge_sets = f[9].split("|") if len(f) > 9 and f[9] else []
                    obj.sample = f[10] if len(f) > 10 else obj.sample
                elif flags & 8:
                    obj.kind = "spinner"
                    obj.end_time = int(float(f[5]))
                    obj.sample = f[6] if len(f) > 6 else obj.sample
                else:
                    raise ValueError(f"Unsupported object type {flags}")
                bm.objects.append(obj)
        except (ValueError, IndexError, TypeError) as exc:
            raise ValueError(f"{source or 'beatmap'}:{line_number}: {exc}") from exc
    bm.timing.sort(key=lambda p: p.time)
    bm.validate()
    return bm


def read(path: str | Path) -> Beatmap:
    path = Path(path)
    data = path.read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1252")
    return parse(text, str(path.resolve()))


def num(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".") if value else "0"


def dumps(bm: Beatmap) -> str:
    bm.validate()
    lines = ["osu file format v14", ""]
    for section in ("General", "Metadata", "Difficulty"):
        lines += [f"[{section}]"] + [f"{k}:{v}" for k, v in getattr(bm, section.lower()).items()] + [""]
    lines += ["[Events]"] + [f"2,{a},{b}" for a, b in bm.breaks] + ["", "[TimingPoints]"]
    for p in sorted(bm.timing, key=lambda p: p.time):
        lines.append(",".join(map(str, [num(p.time), num(p.beat_length), p.meter, p.sample_set, p.sample_index, p.volume, int(p.uninherited), p.effects])))
    lines += ["", "[HitObjects]"]
    for o in bm.objects:
        flags = {"circle": 1, "slider": 2, "spinner": 8}[o.kind] | (4 if o.new_combo else 0) | (o.combo_skip << 4)
        f = [str(round(o.x)), str(round(o.y)), str(o.time), str(flags), str(o.hitsound)]
        if o.kind == "slider":
            f += [o.curve + "|" + "|".join(f"{round(x)}:{round(y)}" for x, y in o.points), str(o.repeats), num(o.length), "|".join(map(str, o.edge_sounds or [0] * (o.repeats + 1))), "|".join(o.edge_sets or ["0:0"] * (o.repeats + 1))]
        elif o.kind == "spinner":
            f += [str(o.end_time)]
        lines.append(",".join(f + [o.sample]))
    return "\n".join(lines) + "\n"


def write(bm: Beatmap, path: str | Path) -> None:
    Path(path).write_text(dumps(bm), encoding="utf-8")

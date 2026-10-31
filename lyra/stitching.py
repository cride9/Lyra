"""Known-token replay through the production section/commit pipeline, without a model."""
import copy
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from . import mapio
from .config import ModelConfig
from .generation import GenerationConfig, candidate
from .io import atomic_json, load_json, parse_map, readonly
from .tokenizer import Tokenizer


class ReplayTokens:
    def __init__(self, source):
        self.source, self.tok = source, Tokenizer()
        self.sections, self.grammar_errors, self.geometry_errors = [], [], []
        self.grammar_error_count = 0

    def prepare(self, ctx, max_tokens, bm):
        records = self.tok.records(self.source, ctx['base'], ctx['start'], ctx['end'])
        budget, selected = max_tokens - len(ctx['prefix']) - 32, []
        for _, group in itertools.groupby(records, key=lambda r: r[0]):
            group = list(group)
            cost = sum(len(r) for _, r in group)
            if cost > budget:
                break
            selected.extend(group); budget -= cost
        if records and not selected:
            raise ValueError('A complete simultaneous-event group exceeds the replay token budget.')
        self.exhausted = len(selected) < len(records)
        self.pending = iter([t for _, record in selected for t in record] + [self.tok.ids['EOS']])
        self.sections.append(dict(start=ctx['start'], end=ctx['end'], base=ctx['base'],
                                  prefix_tokens=len(ctx['prefix']), events=len(selected),
                                  last_event=selected[-1][0] if selected else None,
                                  before=dict(objects=len(bm.objects), breaks=len(bm.breaks),
                                              last_time=bm.objects[-1].time if bm.objects else None),
                                  shortened=self.exhausted))

    def finish(self, bm, end):
        self.sections[-1].update(next_start=end, after=dict(objects=len(bm.objects), breaks=len(bm.breaks),
                                        last_time=bm.objects[-1].time if bm.objects else None))

    def next(self, grammar, allowed):
        token = next(self.pending)
        if token not in allowed:
            self.grammar_error_count += 1
            if len(self.grammar_errors) < 100:
                self.grammar_errors.append(dict(section_index=len(self.sections)-1, state=grammar.state,
                                                time=grammar.at, token=self.tok.names[token]))
        return token


def event_tokens(bm):
    """Absolute event identity, independent of any section origin."""
    tok = Tokenizer()
    records = [(o.time, tok.object(bm, o, o.time)) for o in bm.objects]
    records += [(a, [tok.ids['BREAK'], *tok.time(a, a), *tok.integer(b-a), tok.ids['END']])
                for a, b in bm.breaks]
    return sorted(records, key=lambda r: r[0])


def canonical_map(source):
    # No windowing or production assembly here: the existing decoder is the oracle.
    bm = source.copy()
    bm.objects, bm.breaks = [], []
    bm.timing = copy.deepcopy([p for p in source.timing if p.uninherited])
    tok = Tokenizer()
    for at, record in event_tokens(source):
        tok.decode(record, bm, at)
    return bm


def compare(source, restored):
    canonical = canonical_map(source)
    expected, actual = event_tokens(canonical), event_tokens(restored)
    first = next((dict(kind='event', index=i, expected=a, actual=b)
                  for i, (a, b) in enumerate(itertools.zip_longest(expected, actual)) if a != b), None)
    timing_expected, timing_actual = [asdict(p) for p in canonical.timing], [asdict(p) for p in restored.timing]
    timing_difference = next((dict(kind='timing', index=i, expected=a, actual=b)
                  for i, (a, b) in enumerate(itertools.zip_longest(timing_expected, timing_actual)) if a != b), None)
    first = first or timing_difference
    starts = [b.time-a.time for a, b in zip(source.objects, restored.objects)]
    ends = [abs(b.time+canonical.duration(b)-a.time-source.duration(a))
            for a, b in zip(source.objects, canonical.objects)]
    return dict(passed=first is None and len(source.objects) == len(restored.objects),
                first_difference=first, source_objects=len(source.objects), restored_objects=len(restored.objects),
                source_breaks=len(source.breaks), restored_breaks=len(restored.breaks),
                canonical_timing_equal=timing_difference is None,
                start_time_mismatches=sum(d != 0 for d in starts),
                source_quantization=dict(max_end_error_ms=max(ends, default=0),
                    max_coordinate_error=max((max(abs(a.x-b.x), abs(a.y-b.y))
                                              for a, b in zip(source.objects, canonical.objects)), default=0)))


def replay_map(source, duration, config=None, progress=lambda _: None, cancelled=lambda: False):
    config = config or ModelConfig()
    replay = ReplayTokens(source)
    settings = {k: float(source.difficulty[name]) for k, name in
                dict(AR='ApproachRate', OD='OverallDifficulty', CS='CircleSize', HP='HPDrainRate').items()}
    restored = candidate(SimpleNamespace(config=config), None, duration,
                         [p for p in source.timing if p.uninherited], GenerationConfig(), settings,
                         torch.device('cpu'), 42, progress, cancelled, replay=replay, template=source)
    result = compare(source, restored)
    result.update(sections=replay.sections, grammar_rejections=replay.grammar_errors,
                  grammar_rejection_count=replay.grammar_error_count,
                  geometry_rejections=replay.geometry_errors, model_context=asdict(config))
    if result['first_difference']:
        difference = result['first_difference']
        value = difference['expected'] or difference['actual']
        at = max(0, value['time']) if difference['kind'] == 'timing' else value[0]
        result['first_difference']['section'] = next((s for s in replay.sections
            if s['last_event'] is not None and s['start'] <= at <= s['last_event']), None)
    return restored, result


def band_index(stars):
    return sum(stars >= limit for limit in (2, 2.7, 4, 5.3, 6.5))


def verify_stitching(dataset_path, output, beatmap_id=4370690, per_band=4,
                     progress=print, cancelled=lambda: False):
    if per_band < 0:
        raise ValueError('per_band must not be negative.')
    manifest, root = load_json(dataset_path), Path(output)
    selected, buckets = [], [[] for _ in range(6)]
    with readonly(manifest['index']) as db:
        for (raw,) in db.execute('SELECT payload FROM maps ORDER BY id'):
            row = json.loads(raw)
            if Path(row['map_path']).stem == str(beatmap_id):
                selected.append(row)
            else:
                rank = hashlib.sha256(('42:'+row['id']).encode()).hexdigest()
                buckets[band_index(row['stars'])].append((rank, row))
    if not selected:
        raise ValueError(f'Beatmap {beatmap_id} was not found in the frozen dataset.')
    selected += [row for bucket in buckets for _, row in sorted(bucket, key=lambda r: r[0])[:per_band]]
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for number, row in enumerate(selected, 1):
        if cancelled():
            raise InterruptedError('Stitching verification stopped.')
        progress(f'Verify {number}/{len(selected)}: {row["map_path"]}')
        folder = root / Path(row['map_path']).stem
        try:
            source = parse_map(row['map_path'])
            restored, result = replay_map(source, row['duration_ms'], cancelled=cancelled)
            folder.mkdir(parents=True, exist_ok=True)
            mapio.write(restored, folder / 'reconstructed.osu')
            result.update(map_id=row['id'], source=row['map_path'], stars=row['stars'])
        except InterruptedError:
            raise
        except Exception as exc:
            result = dict(passed=False, map_id=row['id'], source=row['map_path'], error=str(exc))
        atomic_json(folder / 'report.json', result)
        results.append(result)
        atomic_json(root / 'report.json', dict(passed=all(r['passed'] for r in results),
                    complete=len(results) == len(selected), maps=results))
    return dict(passed=all(r['passed'] for r in results), count=len(results), report=str(root / 'report.json'))

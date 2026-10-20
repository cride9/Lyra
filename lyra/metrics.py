"""Append-only update history, queryable after any browser/server restart."""
import json
from pathlib import Path
import sqlite3
import math
import csv
import io


class Metrics:
    def __init__(self, run):
        self.path = Path(run) / 'metrics.sqlite'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS metrics(step INTEGER PRIMARY KEY, payload TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS archived_metrics(id INTEGER PRIMARY KEY,step INTEGER,restored_step INTEGER,payload TEXT)')

    def write(self, value):
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute('INSERT OR REPLACE INTO metrics VALUES(?,?)', (value['step'], json.dumps(value, allow_nan=False)))

    def truncate_after(self, step):
        # Keep abandoned updates for statistics/audit if a machine crash forced
        # a rollback to an older checkpoint. Only the active graph is rewound.
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute('INSERT INTO archived_metrics(step,restored_step,payload) SELECT step,?,payload FROM metrics WHERE step>?', (step, step))
            db.execute('DELETE FROM metrics WHERE step>?', (step,))

    def recent_speed(self, count=50):
        with sqlite3.connect(self.path, timeout=30) as db:
            rows = db.execute("SELECT json_extract(payload,'$.seconds_per_update') FROM metrics ORDER BY step DESC LIMIT ?", (count,)).fetchall()
        values = [r[0] for r in rows if isinstance(r[0], (int, float)) and r[0] > 0]
        return sum(values) / len(values) if values else None

    def latest(self):
        with sqlite3.connect(self.path, timeout=30) as db:
            row = db.execute('SELECT payload FROM metrics ORDER BY step DESC LIMIT 1').fetchone()
        return json.loads(row[0]) if row else {}

    def records(self, include_archived=False):
        db = sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True, timeout=30, check_same_thread=False)
        try:
            if include_archived:
                query = """SELECT kind,payload,restored_step FROM (
                    SELECT 'update' AS kind,payload,NULL AS restored_step FROM metrics
                    UNION ALL SELECT 'rolled_back_update',payload,restored_step FROM archived_metrics)
                    ORDER BY json_extract(payload,'$.timestamp_utc'),json_extract(payload,'$.step')"""
                for kind, payload, restored in db.execute(query):
                    yield dict(record_type=kind, restored_step=restored, **json.loads(payload))
            else:
                for (payload,) in db.execute('SELECT payload FROM metrics ORDER BY step'):
                    yield dict(record_type='update', **json.loads(payload))
        finally:
            db.close()

    def export(self, format='csv', include_archived=False):
        if format == 'jsonl':
            for value in self.records(include_archived):
                yield json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n'
            return
        if format != 'csv':
            raise ValueError('Choose CSV or JSONL export.')
        columns = ['record_type', 'restored_step', 'timestamp_utc', 'step', 'target_steps',
                   'elapsed_seconds', 'loss', 'loss_ema', 'token_loss', 'beat_loss',
                   'validation_token_loss', 'validation.beat_loss', 'validation.windows',
                   'lr', 'gradient_norm', 'grad_scale', 'amp_retries', 'seconds_per_update',
                   'data_seconds', 'training_seconds', 'data_wait_fraction',
                   'target_tokens', 'target_tokens_per_second', 'examples_per_second',
                   'batch_size', 'effective_batch', 'visited_windows', 'completed_cycles',
                   'cycle_fraction', 'train_windows', 'parameters', 'peak_vram_gib',
                   'reserved_vram_gib', 'free_vram_gib',
                   *['cache.' + k for k in ('ram_hits', 'disk_hits', 'reused_v2', 'misses', 'writes',
                                            'audio_decodes', 'errors', 'evictions', 'bypasses')]]
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=columns)
        yield '\ufeff'  # UTF-8 BOM for spreadsheet import.
        writer.writeheader(); yield buffer.getvalue(); buffer.seek(0); buffer.truncate(0)
        for value in self.records(include_archived):
            flat = dict(value)
            for section in ('validation', 'cache'):
                flat.update({section + '.' + k: v for k, v in value.get(section, {}).items()})
            writer.writerow({k: flat.get(k) for k in columns})
            yield buffer.getvalue(); buffer.seek(0); buffer.truncate(0)

    def history(self, points=600):
        with sqlite3.connect(self.path, timeout=30) as db:
            n = db.execute('SELECT COUNT(*) FROM metrics').fetchone()[0]
            stride = max(1, math.ceil(n / max(1, points)))
            rows = db.execute('SELECT payload FROM metrics WHERE step % ?=0 OR step=(SELECT MAX(step) FROM metrics) ORDER BY step', (stride,)).fetchall()
        values = [json.loads(r[0]) for r in rows]
        return [{k: v.get(k) for k in ('step', 'loss', 'token_loss', 'beat_loss', 'validation_token_loss', 'elapsed_seconds')} for v in values]

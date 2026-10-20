"""Bounded RAM + NVMe array cache; optional read-only reuse of V2 mel tiles."""
from collections import OrderedDict, Counter
from contextlib import contextmanager
import io
import json
import math
from pathlib import Path
import shutil
import sqlite3
import time
import uuid
import zlib
import numpy as np
from filelock import FileLock
from .io import identity, readonly


class ArrayCache:
    def __init__(self, root, gb=180, ram_gib=1.5, min_free_gib=8, legacy=None):
        if not math.isfinite(gb) or gb <= 0 or ram_gib < 0 or min_free_gib < 0:
            raise ValueError('Érvénytelen cache-keret.')
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity, self.reserve = int(gb * 10 ** 9), int(min_free_gib * 1024 ** 3)
        self.ram_capacity = int(ram_gib * 1024 ** 3)
        self.ram, self.ram_bytes, self.stats = OrderedDict(), 0, Counter()
        self.legacy = Path(legacy) if legacy else None
        self.lock = FileLock(str(self.root / 'budget.lock'), timeout=120)
        with self.lock, self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS entries(key TEXT PRIMARY KEY, file TEXT, size INTEGER, touched REAL, crc INTEGER)')
            db.execute('CREATE INDEX IF NOT EXISTS lru ON entries(touched)')
            db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value INTEGER)')
            db.execute("INSERT OR IGNORE INTO metadata SELECT 'used',COALESCE(SUM(size),0) FROM entries")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root / 'index.sqlite', timeout=120)
        try:
            with db:
                yield db
        finally:
            db.close()

    def remember(self, key, value):
        size = sum(a.nbytes for a in value.values())
        if size > self.ram_capacity:
            return
        if key in self.ram:
            self.ram_bytes -= sum(a.nbytes for a in self.ram.pop(key).values())
        while self.ram and self.ram_bytes + size > self.ram_capacity:
            _, old = self.ram.popitem(last=False)
            self.ram_bytes -= sum(a.nbytes for a in old.values())
        for a in value.values():
            a.flags.writeable = False
        self.ram[key] = value
        self.ram_bytes += size

    def get(self, namespace, key):
        full = identity([namespace, key])
        if full in self.ram:
            self.stats['ram_hits'] += 1
            self.ram.move_to_end(full)
            return self.ram[full]
        try:
            with self.lock, self.connect() as db:
                row = db.execute('SELECT file,size,touched,crc FROM entries WHERE key=?', (full,)).fetchone()
                if row:
                    path = self.root / row[0]
                    if path.parent != self.root or not path.name.startswith('entry-'):
                        raise ValueError('Érvénytelen cache-bejegyzés.')
                    try:
                        raw = path.read_bytes()
                        if zlib.crc32(raw) != row[3]:
                            raise ValueError('Sérült cache-bejegyzés.')
                        with np.load(io.BytesIO(raw), allow_pickle=False) as f:
                            value = {k: f[k] for k in f.files}
                        if row[2] < time.time() - 60:
                            db.execute('UPDATE entries SET touched=? WHERE key=?', (time.time(), full))
                    except (OSError, ValueError, EOFError):
                        path.unlink(missing_ok=True)
                        db.execute('DELETE FROM entries WHERE key=?', (full,))
                        db.execute("UPDATE metadata SET value=value-? WHERE key='used'", (row[1],))
                        self.stats['errors'] += 1
                        row = None
                    if row:
                        self.stats['disk_hits'] += 1
                        self.remember(full, value)
                        return value
        except (OSError, sqlite3.DatabaseError):
            self.stats['errors'] += 1
        value = self.borrow(namespace, key)
        if value is not None:
            self.stats['reused_v2'] += 1
            self.remember(full, value)
            return value
        self.stats['misses'] += 1
        return None

    def borrow(self, namespace, key):
        if self.legacy is None or namespace not in ('mel-blocks', 'audio-targets'):
            return None
        index = self.legacy / 'index.sqlite'
        if not index.is_file():
            return None
        import hashlib
        legacy_key = hashlib.sha256((':' + namespace + ':' + key).encode()).hexdigest()
        try:
            with FileLock(str(self.legacy / 'budget.lock'), timeout=120), readonly(index) as db:
                row = db.execute('SELECT folder,arrays FROM entries WHERE key=?', (legacy_key,)).fetchone()
                if row is None:
                    return None
                folder = (self.legacy / row[0]).resolve()
                if folder.parent != self.legacy.resolve() or not folder.name.startswith('entry-'):
                    return None
                result = {}
                for name, desc in json.loads(row[1]).items():
                    if not name.isidentifier():
                        return None
                    path = folder / (name + '.npy')
                    raw = path.read_bytes()
                    if 'crc32' in desc and zlib.crc32(raw) != desc['crc32']:
                        return None
                    value = np.load(io.BytesIO(raw), allow_pickle=False)
                    if list(value.shape) != desc['shape'] or str(value.dtype) != desc['dtype']:
                        return None
                    result[name] = value
                return result
        except (OSError, sqlite3.DatabaseError, ValueError, KeyError):
            return None

    def put(self, namespace, key, value):
        full = identity([namespace, key])
        self.remember(full, value)
        buffer = io.BytesIO()
        np.savez(buffer, **value)  # No CPU-heavy compression in the loading path.
        raw = buffer.getvalue()
        if len(raw) > self.capacity:
            self.stats['bypasses'] += 1
            return
        path = self.root / ('entry-' + uuid.uuid4().hex + '.npz')
        try:
            with self.lock, self.connect() as db:
                if db.execute('SELECT 1 FROM entries WHERE key=?', (full,)).fetchone():
                    return
                used = db.execute("SELECT value FROM metadata WHERE key='used'").fetchone()[0]
                while used + len(raw) > self.capacity or shutil.disk_usage(self.root).free < self.reserve + len(raw) + 1024 ** 2:
                    oldest = db.execute('SELECT key,file,size FROM entries ORDER BY touched LIMIT 1').fetchone()
                    if oldest is None:
                        self.stats['bypasses'] += 1
                        return
                    target = (self.root / oldest[1]).resolve()
                    if target.parent != self.root or not target.name.startswith('entry-'):
                        raise ValueError('Érvénytelen cache-törlési cél.')
                    target.unlink(missing_ok=True)
                    db.execute('DELETE FROM entries WHERE key=?', (oldest[0],))
                    used -= oldest[2]
                    db.execute("UPDATE metadata SET value=? WHERE key='used'", (used,))
                    db.commit()
                    self.stats['evictions'] += 1
                path.write_bytes(raw)
                db.execute('INSERT INTO entries VALUES(?,?,?,?,?)', (full, path.name, len(raw), time.time(), zlib.crc32(raw)))
                db.execute("UPDATE metadata SET value=value+? WHERE key='used'", (len(raw),))
                self.stats['writes'] += 1
        except (OSError, sqlite3.DatabaseError):
            path.unlink(missing_ok=True)
            self.stats['bypasses'] += 1

    def recording_lock(self, key):
        folder = self.root / 'locks'
        folder.mkdir(exist_ok=True)
        return FileLock(str(folder / (identity(key) + '.lock')), timeout=300)

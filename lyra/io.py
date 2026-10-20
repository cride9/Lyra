"""Atomic records, frozen source identities, and bounded archive access."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from urllib.parse import unquote
import zipfile


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temp = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as f:
            temp = Path(f.name)
            f.write(payload); f.flush(); os.fsync(f.fileno())
        for attempt in range(8):
            try:
                temp.replace(path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(.025 * (attempt + 1))
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


def telemetry(path, default=None):
    try:
        value = load_json(path, default)
        return value if isinstance(value, dict) else default
    except (OSError, ValueError, UnicodeError):
        return default


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


@contextmanager
def readonly(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=30)
    try:
        db.execute('PRAGMA query_only=ON')
        yield db
    finally:
        db.close()


def source_parts(ref):
    if not str(ref).startswith('osz://'):
        return Path(ref), None
    archive, separator, member = str(ref)[6:].partition('#')
    if not separator:
        raise ValueError('Hiányos archívumhivatkozás.')
    return Path(unquote(archive)), unquote(member)


def signature(ref):
    path, _ = source_parts(ref)
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


def read_bytes(ref, maximum=256 * 1024 ** 2):
    path, member = source_parts(ref)
    if member is None:
        if path.stat().st_size > maximum:
            raise ValueError('Túl nagy forrásfájl.')
        return path.read_bytes()
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(member)
        if not 0 < info.file_size <= maximum or (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError('Érvénytelen archívumtag.')
        return z.read(info)


def parse_map(ref):
    from . import mapio
    raw = read_bytes(ref, 8 * 1024 ** 2)
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = raw.decode('cp1252')
    return mapio.parse(text, str(ref))

import hashlib
import json
import os
import re
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def slug(model):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', model)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    atomic_text(path, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))

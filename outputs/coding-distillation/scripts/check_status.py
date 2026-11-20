import glob
import json
from pathlib import Path

records = []
for p in Path("outputs/coding-distillation/data-v3/raw/codegemma_7b-instruct-fp16/config_b").glob("*/record.json"):
    try:
        records.append(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        pass

accepted = sum(1 for r in records if r.get("accepted"))
total = len(records)
print(f"CodeGemma 7B FP16 Progress: {total}/1000 evaluated")
print(f"Accepted: {accepted}/{total} ({accepted/total*100:.2f}%)" if total else "No records")

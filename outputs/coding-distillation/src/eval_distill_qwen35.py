"""
Small held-out evaluation and Ollama smoke-test harness comparing the base
Qwen3.5-4B student against the distilled student (test-model-distilled) using
identical Ollama generation settings, identical non-thinking mode, Docker
sandbox verification, and the project's explanation rubric.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml

from .explanation import verify_explanation
from .sandbox_client import DockerValidator
from .validate import extract


def extract_explanation_and_code(raw: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Split a model response into (explanation_text, extracted_code, error_reason).
    Extracts the last ```python code block if present, or falls back to validate.extract.
    """
    fences = list(re.finditer(r"```([^\n]*)\n(.*?)```", raw, re.S))
    py_fences = [
        m for m in fences if m.group(1).strip().lower() in ("", "python", "python3", "py")
    ]
    if not py_fences:
        return raw.strip(), None, "no_python_code_block"

    last = py_fences[-1]
    explanation = raw[: last.start()].strip()
    code_raw = last.group(2)
    try:
        code = extract(f"```python\n{code_raw}```")
    except Exception as e:
        return explanation, None, f"code_extract_error:{e}"
    return explanation, code, None


def ollama_unload(base_url: str, model: str) -> None:
    try:
        httpx.post(
            f"{base_url}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30.0,
        )
    except Exception:
        pass


def ollama_chat(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    num_predict: int = 1536,
    num_ctx: int = 4096,
    temperature: float = 0.0,
    seed: int = 42,
    think: bool = False,
    timeout: float = 180.0,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "stream": False,
        "think": think,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
        "keep_alive": "5m",
    }
    t0 = time.perf_counter()
    resp = httpx.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    wall = round(time.perf_counter() - t0, 2)
    data = resp.json()
    msg = data.get("message", {})
    content = msg.get("content", "")
    thinking = msg.get("thinking", "")
    eval_count = data.get("eval_count", 0)
    eval_dur_ns = data.get("eval_duration", 0)
    tps = round(eval_count / (eval_dur_ns / 1e9), 2) if eval_dur_ns > 0 else 0.0
    return {
        "content": content,
        "thinking": thinking,
        "wall_seconds": wall,
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "eval_count": eval_count,
        "tokens_per_sec": tps,
        "done_reason": data.get("done_reason"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate distilled vs base Qwen3.5-4B on held-out coding subset.")
    parser.add_argument("--config", type=Path, default=Path("config-v3.yaml"))
    parser.add_argument(
        "--heldout",
        type=Path,
        default=Path("data-v3/distillation/heldout_eval_subset.jsonl"),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data-v3/problems/verified_corpus.jsonl"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Ollama model names to evaluate under identical settings.",
    )
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--num-predict", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results-v3/distillation/qwen35_4b_config_b/heldout_eval_report.json"),
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base_url = cfg["ollama"]["base_url"]
    validator = DockerValidator(cfg["validation"])

    corpus = {}
    with args.corpus.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                corpus[p["id"]] = p

    eval_items = []
    with args.heldout.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_items.append(json.loads(line))
    eval_items = eval_items[: args.limit]

    # Assert zero overlap with training dataset
    train_path = Path("data-v3/distillation/qwen2.5_7b_config_b_sft.jsonl")
    if train_path.exists():
        with train_path.open("r", encoding="utf-8") as f:
            train_ids = {json.loads(line)["problem_id"] for line in f if line.strip()}
        overlap = [it["problem_id"] for it in eval_items if it["problem_id"] in train_ids]
        if overlap:
            raise RuntimeError(f"Held-out evaluation leaked training IDs: {overlap}")

    report: Dict[str, Any] = {
        "eval_settings": {
            "heldout_count": len(eval_items),
            "think": False,
            "temperature": args.temperature,
            "seed": args.seed,
            "num_ctx": 4096,
            "num_predict": args.num_predict,
        },
        "models": {},
    }

    for model in args.models:
        print(f"\n=== Evaluating {model} on {len(eval_items)} held-out problems ===", flush=True)
        # Ensure clean GPU before starting this model
        for other in args.models:
            ollama_unload(base_url, other)

        per_problem = []
        code_passed = 0
        expl_passed = 0
        both_passed = 0
        total_tps = []
        total_wall = []

        for idx, item in enumerate(eval_items, 1):
            pid = item["problem_id"]
            problem = corpus[pid]
            gen = ollama_chat(
                base_url=base_url,
                model=model,
                system_prompt=item["system"],
                user_prompt=item["prompt"],
                num_predict=args.num_predict,
                temperature=args.temperature,
                seed=args.seed,
                think=False,
            )
            explanation, code, ext_err = extract_explanation_and_code(gen["content"])

            if code is not None:
                val = validator.validate(code, problem)
                c_ok = bool(val.get("accepted"))
                c_status = val.get("status", "unknown")
                e_check = verify_explanation(explanation, code)
                e_ok = bool(e_check.get("verified"))
                e_reason = e_check.get("reason", "ok")
            else:
                c_ok = False
                c_status = ext_err or "extraction_error"
                e_ok = False
                e_reason = "no_code_extracted"

            if c_ok:
                code_passed += 1
            if e_ok:
                expl_passed += 1
            if c_ok and e_ok:
                both_passed += 1
            total_tps.append(gen["tokens_per_sec"])
            total_wall.append(gen["wall_seconds"])

            rec = {
                "problem_id": pid,
                "source": item["source"],
                "fn_name": item.get("fn_name"),
                "code_passed": c_ok,
                "code_status": c_status,
                "explanation_passed": e_ok,
                "explanation_reason": e_reason,
                "both_passed": c_ok and e_ok,
                "eval_tokens": gen["eval_count"],
                "tokens_per_sec": gen["tokens_per_sec"],
                "wall_seconds": gen["wall_seconds"],
                "done_reason": gen["done_reason"],
                "has_thinking_output": bool(gen["thinking"].strip()),
            }
            per_problem.append(rec)
            print(
                f"  [{idx:02d}/{len(eval_items)}] {pid[:10]} ({item['source']:8s} {item.get('fn_name',''):22s}) "
                f"code={str(c_ok):5s} ({c_status:18s}) expl={str(e_ok):5s} "
                f"toks={gen['eval_count']:4d} tps={gen['tokens_per_sec']:5.1f}",
                flush=True,
            )

        ollama_unload(base_url, model)
        n = len(eval_items)
        report["models"][model] = {
            "code_passed": code_passed,
            "code_pass_rate": round(100.0 * code_passed / n, 1),
            "explanation_passed": expl_passed,
            "explanation_pass_rate": round(100.0 * expl_passed / n, 1),
            "both_passed": both_passed,
            "both_pass_rate": round(100.0 * both_passed / n, 1),
            "mean_tokens_per_sec": round(sum(total_tps) / n, 2),
            "mean_wall_seconds": round(sum(total_wall) / n, 2),
            "per_problem": per_problem,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== HELD-OUT EVALUATION SUMMARY (saved to {args.output}) ===", flush=True)
    for model, stats in report["models"].items():
        print(
            f"  {model:28s}: code={stats['code_passed']}/{len(eval_items)} ({stats['code_pass_rate']}%), "
            f"expl={stats['explanation_passed']}/{len(eval_items)} ({stats['explanation_pass_rate']}%), "
            f"both={stats['both_passed']}/{len(eval_items)} ({stats['both_pass_rate']}%), "
            f"avg_tps={stats['mean_tokens_per_sec']}",
            flush=True,
        )


if __name__ == "__main__":
    main()

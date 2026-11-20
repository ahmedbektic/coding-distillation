"""
Prepare a clean, deduplicated, training-compatible SFT dataset for offline
sequence-level knowledge distillation from existing accepted Configuration B
examples of qwen2.5:7b-instruct-fp16.

Preserves all existing datasets and never regenerates teacher responses.
"""

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .repair import extract_function_signature


STUDENT_SYSTEM_PROMPT = (
    "You are an expert Python algorithm tutor and software engineer. "
    "Explain your algorithm clearly and provide clean, efficient Python 3 "
    "solutions using only the Python standard library."
)


def build_student_prompt(problem: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """
    Build the clean student input prompt containing ONLY:
    - Clear instructions matching the target (explanation + verified Python code)
    - Required function signature (where available)
    - Original programming problem statement

    Never includes reference solutions, hidden test cases, debugging feedback,
    or failed attempts.
    """
    question = (problem.get("question") or "").strip()
    starter_code = (problem.get("starter_code") or "").strip()
    sig_str, _ = extract_function_signature(problem)

    # Prefer the exact single-line def signature from starter_code if present,
    # otherwise use the AST-extracted signature (name + parameter names only).
    if starter_code and starter_code.startswith("def "):
        display_sig = starter_code.splitlines()[0].rstrip()
        if not display_sig.endswith(":"):
            display_sig = sig_str or display_sig
    else:
        display_sig = sig_str

    parts = [
        "Solve the following Python 3 programming problem using only the Python standard library.",
        (
            "First provide a clear algorithm explanation covering:\n"
            "1. Algorithm selection and overall approach\n"
            "2. Important steps and data structures used\n"
            "3. Justification for correctness and boundary conditions\n"
            "4. Time complexity and auxiliary space complexity\n\n"
            "Then provide the complete Python 3 implementation inside a single ```python code block. "
            "Return the answer directly; do not read from stdin."
        ),
    ]

    if display_sig:
        parts.append(
            "Required function signature (preserve this exact signature):\n"
            f"```python\n{display_sig}\n    pass\n```"
        )

    parts.append(f"Problem:\n{question}")
    return "\n\n".join(parts), display_sig


def build_target_completion(explanation: str, code: str) -> str:
    """Format the existing teacher explanation followed by the verified Python code."""
    expl_clean = explanation.strip()
    code_clean = code.rstrip("\r\n") + "\n"
    return f"{expl_clean}\n\n```python\n{code_clean}```"


def load_corpus(corpus_path: Path) -> Dict[str, Dict[str, Any]]:
    corpus: Dict[str, Dict[str, Any]] = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            corpus[item["id"]] = item
    return corpus


def collect_config_b_examples(
    data_dir: Path,
    teacher_slug: str = "qwen2.5_7b-instruct-fp16",
    include_raw_accepted: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load accepted Configuration B examples for the given teacher, deduplicating
    strictly by problem_id. Reads the verified export first and optionally
    reconciles any additional accepted raw records (e.g. consensus retries not
    yet re-exported), without mutating existing files.
    """
    corpus_path = data_dir / "problems" / "verified_corpus.jsonl"
    exported_path = data_dir / "verified" / f"{teacher_slug}_config_b.jsonl"
    raw_dir = data_dir / "raw" / teacher_slug / "config_b"

    corpus = load_corpus(corpus_path)

    seen_ids = set()
    duplicate_ids: List[str] = []
    excluded_records: List[Dict[str, str]] = []
    eligible: List[Dict[str, Any]] = []
    from_exported = 0
    from_raw_only = 0
    attempts_distribution: Dict[str, int] = {}

    def _process_record(
        pid: str,
        explanation: str,
        code: str,
        origin_file: str,
        attempts_count: Optional[int],
        consensus_retry: bool,
    ) -> None:
        nonlocal from_exported, from_raw_only
        if pid in seen_ids:
            duplicate_ids.append(pid)
            return

        problem = corpus.get(pid)
        if problem is None:
            excluded_records.append({"problem_id": pid, "reason": "missing_in_verified_corpus"})
            return

        question = (problem.get("question") or "").strip()
        expl_clean = (explanation or "").strip()
        code_clean = (code or "").strip()

        if not question:
            excluded_records.append({"problem_id": pid, "reason": "empty_question"})
            return
        if not expl_clean:
            excluded_records.append({"problem_id": pid, "reason": "empty_explanation"})
            return
        if not code_clean:
            excluded_records.append({"problem_id": pid, "reason": "empty_code"})
            return

        student_prompt, signature = build_student_prompt(problem)
        target_completion = build_target_completion(expl_clean, code)

        # Safety sanity check: ensure no feedback/retry leakage in student prompt
        leak_markers = ["Test Failure (wrong_answer)", "Actual output:", "previous_code"]
        for marker in leak_markers:
            if marker in student_prompt:
                excluded_records.append({"problem_id": pid, "reason": f"leak_marker:{marker}"})
                return

        seen_ids.add(pid)
        if origin_file == "verified_export":
            from_exported += 1
        else:
            from_raw_only += 1

        att_key = str(attempts_count) if attempts_count is not None else "unknown"
        attempts_distribution[att_key] = attempts_distribution.get(att_key, 0) + 1

        eligible.append(
            {
                "problem_id": pid,
                "source": problem.get("source", "unknown"),
                "difficulty": problem.get("difficulty", "unknown"),
                "signature": signature,
                "attempts_count": attempts_count,
                "consensus_retry": consensus_retry,
                "origin": origin_file,
                "system": STUDENT_SYSTEM_PROMPT,
                "prompt": student_prompt,
                "completion": target_completion,
                "messages": [
                    {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                    {"role": "user", "content": student_prompt},
                    {"role": "assistant", "content": target_completion},
                ],
            }
        )

    # 1. Load verified export records (primary source)
    if exported_path.exists():
        with exported_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("configuration") != "independent_solution_distillation":
                    excluded_records.append(
                        {"problem_id": rec.get("problem_id", "unknown"), "reason": "non_config_b"}
                    )
                    continue
                pid = rec["problem_id"]
                raw_path = raw_dir / pid / "record.json"
                attempts_count = None
                consensus_retry = False
                if raw_path.exists():
                    with raw_path.open("r", encoding="utf-8") as rf:
                        raw_rec = json.load(rf)
                    if not raw_rec.get("accepted") or not raw_rec.get("validation", {}).get("accepted"):
                        excluded_records.append({"problem_id": pid, "reason": "raw_not_accepted"})
                        continue
                    attempts_count = raw_rec.get("attempts_count", len(raw_rec.get("attempts", [])))
                    consensus_retry = bool(raw_rec.get("consensus_retry", False))
                _process_record(
                    pid=pid,
                    explanation=rec.get("explanation", ""),
                    code=rec.get("code", ""),
                    origin_file="verified_export",
                    attempts_count=attempts_count,
                    consensus_retry=consensus_retry,
                )

    # 2. Reconcile any additional accepted raw Config B records (e.g. feedback-guided consensus retries)
    if include_raw_accepted and raw_dir.exists():
        for pid in sorted(os.listdir(raw_dir)):
            if pid in seen_ids:
                continue
            raw_path = raw_dir / pid / "record.json"
            if not raw_path.exists():
                continue
            with raw_path.open("r", encoding="utf-8") as rf:
                raw_rec = json.load(rf)
            if raw_rec.get("status") != "accepted":
                continue
            if not raw_rec.get("accepted") or not raw_rec.get("validation", {}).get("accepted"):
                excluded_records.append({"problem_id": pid, "reason": "raw_validation_not_accepted"})
                continue
            if not raw_rec.get("explanation_verification", {}).get("verified"):
                excluded_records.append({"problem_id": pid, "reason": "raw_explanation_not_verified"})
                continue
            _process_record(
                pid=pid,
                explanation=raw_rec.get("explanation", ""),
                code=raw_rec.get("extracted_code", ""),
                origin_file="raw_accepted_retry",
                attempts_count=raw_rec.get("attempts_count", len(raw_rec.get("attempts", []))),
                consensus_retry=bool(raw_rec.get("consensus_retry", False)),
            )

    # Sort deterministically by problem_id for reproducibility
    eligible.sort(key=lambda r: r["problem_id"])

    summary = {
        "teacher_slug": teacher_slug,
        "configuration": "config_b",
        "eligible_count": len(eligible),
        "from_verified_export": from_exported,
        "from_raw_accepted_only": from_raw_only,
        "duplicate_count": len(duplicate_ids),
        "duplicate_ids": duplicate_ids,
        "excluded_count": len(excluded_records),
        "excluded_records": excluded_records,
        "sources": {
            s: sum(1 for r in eligible if r["source"] == s)
            for s in sorted({r["source"] for r in eligible})
        },
        "signatures_present": sum(1 for r in eligible if r["signature"]),
        "attempts_distribution": dict(sorted(attempts_distribution.items())),
    }
    return eligible, summary


def build_heldout_eval_subset(
    data_dir: Path,
    train_ids: set,
    n_mbpp: int = 6,
    n_codewars: int = 6,
) -> List[Dict[str, Any]]:
    """
    Select a small, deterministic, strictly held-out evaluation subset from
    verified_corpus.jsonl that has ZERO overlap with train_ids. Prefers problems
    independently solved by another teacher (deepseek-coder) so we know they are
    clean, solvable, and have working Docker test harnesses.
    """
    corpus = load_corpus(data_dir / "problems" / "verified_corpus.jsonl")
    ds_raw = data_dir / "raw" / "deepseek-coder_6.7b-instruct-fp16" / "config_b"

    mbpp_candidates = []
    cw_candidates = []
    for pid in sorted(corpus.keys()):
        if pid in train_ids:
            continue
        p = corpus[pid]
        rp = ds_raw / pid / "record.json"
        if not rp.exists():
            continue
        with rp.open("r", encoding="utf-8") as f:
            rec = json.load(f)
        if rec.get("status") != "accepted":
            continue
        if p.get("source") == "mbpp":
            mbpp_candidates.append(p)
        elif p.get("source") == "codewars":
            cw_candidates.append(p)

    selected = mbpp_candidates[:n_mbpp] + cw_candidates[:n_codewars]
    eval_rows = []
    for p in selected:
        student_prompt, sig = build_student_prompt(p)
        eval_rows.append(
            {
                "problem_id": p["id"],
                "source": p["source"],
                "difficulty": p.get("difficulty"),
                "fn_name": p.get("fn_name"),
                "signature": sig,
                "system": STUDENT_SYSTEM_PROMPT,
                "prompt": student_prompt,
            }
        )
    return eval_rows


def compute_token_stats(records: List[Dict[str, Any]], tokenizer_name: str) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    full_lengths = []
    prompt_lengths = []
    completion_lengths = []
    max_record = None

    for r in records:
        prompt_text = tokenizer.apply_chat_template(
            r["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = tokenizer.apply_chat_template(
            r["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        f_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        p_len = len(p_ids)
        f_len = len(f_ids)
        c_len = f_len - p_len
        prompt_lengths.append(p_len)
        full_lengths.append(f_len)
        completion_lengths.append(c_len)
        if max_record is None or f_len > max_record["full_tokens"]:
            max_record = {
                "problem_id": r["problem_id"],
                "source": r["source"],
                "prompt_tokens": p_len,
                "completion_tokens": c_len,
                "full_tokens": f_len,
            }

    def _dist(vals: List[int]) -> Dict[str, Any]:
        s = sorted(vals)
        n = len(s)
        return {
            "count": n,
            "min": s[0],
            "p25": s[int(0.25 * (n - 1))],
            "median": s[int(0.50 * (n - 1))],
            "mean": round(statistics.mean(s), 1),
            "p90": s[int(0.90 * (n - 1))],
            "p95": s[int(0.95 * (n - 1))],
            "p99": s[int(0.99 * (n - 1))],
            "max": s[-1],
            "gt_1024": sum(1 for v in s if v > 1024),
            "gt_1536": sum(1 for v in s if v > 1536),
            "gt_2048": sum(1 for v in s if v > 2048),
            "gt_2560": sum(1 for v in s if v > 2560),
            "gt_3072": sum(1 for v in s if v > 3072),
            "gt_4096": sum(1 for v in s if v > 4096),
        }

    return {
        "tokenizer": tokenizer_name,
        "prompt_tokens": _dist(prompt_lengths),
        "completion_tokens": _dist(completion_lengths),
        "full_tokens": _dist(full_lengths),
        "total_training_tokens_1_epoch": sum(full_lengths),
        "total_completion_tokens_1_epoch": sum(completion_lengths),
        "longest_example": max_record,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Config B SFT dataset for Qwen3.5-4B distillation.")
    parser.add_argument("--data-dir", type=Path, default=Path("data-v3"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data-v3/distillation"),
    )
    parser.add_argument("--teacher-slug", default="qwen2.5_7b-instruct-fp16")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional HF tokenizer ID (e.g. Qwen/Qwen3.5-4B) to compute exact token length stats.",
    )
    parser.add_argument("--heldout-mbpp", type=int, default=6)
    parser.add_argument("--heldout-codewars", type=int, default=6)
    args = parser.parse_args()

    eligible, summary = collect_config_b_examples(
        data_dir=args.data_dir,
        teacher_slug=args.teacher_slug,
        include_raw_accepted=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_path = args.output_dir / "qwen2.5_7b_config_b_sft.jsonl"
    with sft_path.open("w", encoding="utf-8") as f:
        for row in eligible:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    train_ids = {r["problem_id"] for r in eligible}
    heldout = build_heldout_eval_subset(
        data_dir=args.data_dir,
        train_ids=train_ids,
        n_mbpp=args.heldout_mbpp,
        n_codewars=args.heldout_codewars,
    )
    heldout_path = args.output_dir / "heldout_eval_subset.jsonl"
    with heldout_path.open("w", encoding="utf-8") as f:
        for row in heldout:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary["sft_dataset_path"] = str(sft_path)
    summary["heldout_eval_path"] = str(heldout_path)
    summary["heldout_eval_count"] = len(heldout)

    if args.tokenizer:
        summary["token_stats"] = compute_token_stats(eligible, args.tokenizer)

    stats_path = args.output_dir / "qwen2.5_7b_config_b_sft_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

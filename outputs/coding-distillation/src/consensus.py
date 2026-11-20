"""
Cross-Model Consensus Analyzer for Configuration B.
Evaluates how many of the 5 active 7B FP16 models solved each of the 1,000 problems (0..5),
generates detailed distribution statistics, and identifies targeted retry candidates
(problems solved by 3 or 4 models that specific models missed).
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

from .common import jsonl, read_json, slug


ACTIVE_TEACHERS = [
    "qwen2.5:7b-instruct-fp16",
    "qwen2.5-coder:7b-instruct-fp16",
    "deepseek-coder:6.7b-instruct-fp16",
    "codegemma:7b-instruct-fp16",
    "mistral:7b-instruct-v0.3-fp16",
    "codellama:7b-instruct-fp16"
]


def load_model_records(data_dir: Path, teachers: List[str]) -> Dict[str, Dict[str, dict]]:
    """Load all config_b records for the specified teachers keyed by problem_id."""
    records_by_teacher = {}
    for t in teachers:
        t_slug = slug(t)
        t_dir = data_dir / "raw" / t_slug / "config_b"
        teacher_recs = {}
        if t_dir.exists():
            for rf in t_dir.glob("*/record.json"):
                try:
                    rec = read_json(rf)
                    teacher_recs[rec['problem_id']] = rec
                except Exception:
                    pass
        records_by_teacher[t] = teacher_recs
    return records_by_teacher


def analyze_consensus(config: dict, teachers: Optional[List[str]] = None,
                      target_consensus: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Analyzes consensus across all problems in the corpus.
    Returns full analysis dictionary and saves JSON/Markdown reports.
    """
    if teachers is None:
        teachers = config.get('teachers') or ACTIVE_TEACHERS

    if target_consensus is None:
        # Default: all high-consensus levels where >=3 models solved it, but not all models
        target_consensus = list(range(3, len(teachers)))

    data_dir = Path(config['data_dir'])
    results_dir = Path(config.get('results_dir', 'results-v3')) / 'summaries'
    results_dir.mkdir(parents=True, exist_ok=True)

    corpus_file = data_dir / "problems" / "verified_corpus.jsonl"
    corpus = jsonl(corpus_file) if corpus_file.exists() else []
    total_problems = len(corpus)

    model_records = load_model_records(data_dir, teachers)

    # Solve counts and mapping
    consensus_counts = {k: 0 for k in range(len(teachers) + 1)}
    problems_by_consensus = {k: [] for k in range(len(teachers) + 1)}
    solvers_per_problem = {}
    
    # Model individual solve counts
    model_solves = {t: sum(1 for r in model_records[t].values() if r.get('accepted')) for t in teachers}
    model_evaluated = {t: len(model_records[t]) for t in teachers}

    # Breakdown by difficulty and source
    diff_breakdown = {k: defaultdict(int) for k in range(len(teachers) + 1)}
    source_breakdown = {k: defaultdict(int) for k in range(len(teachers) + 1)}

    # Missed problems for targeted consensus levels (e.g. 3, 4, 5)
    targeted_retries = {t: [] for t in teachers}

    for p in corpus:
        pid = p['id']
        solvers = [t for t in teachers if model_records[t].get(pid, {}).get('accepted', False)]
        k = len(solvers)
        consensus_counts[k] += 1
        problems_by_consensus[k].append(pid)
        solvers_per_problem[pid] = solvers

        diff = p.get('difficulty', 'UNKNOWN').upper()
        diff_breakdown[k][diff] += 1

        src = p.get('source', 'unknown')
        source_breakdown[k][src] += 1

        if k in target_consensus:
            for t in teachers:
                if t not in solvers:
                    targeted_retries[t].append({
                        'problem_id': pid,
                        'consensus_count': k,
                        'solvers': solvers,
                        'current_status': model_records[t].get(pid, {}).get('status', 'unattempted'),
                        'difficulty': diff,
                        'source': src
                    })

    target_total = sum(consensus_counts.get(k, 0) for k in target_consensus)
    summary = {
        'teachers': teachers,
        'target_consensus': target_consensus,
        'total_problems': total_problems,
        'consensus_counts': consensus_counts,
        'consensus_percentages': {k: (v / total_problems * 100) if total_problems else 0 for k, v in consensus_counts.items()},
        'model_evaluated': model_evaluated,
        'model_solves': model_solves,
        'model_pass_rates': {t: (model_solves[t] / total_problems * 100) if total_problems else 0 for t in teachers},
        'problems_targeted_consensus_total': target_total,
        'problems_solved_by_3_or_4_total': consensus_counts.get(3, 0) + consensus_counts.get(4, 0),
        'targeted_retry_counts': {t: len(targeted_retries[t]) for t in teachers},
        'targeted_retries': targeted_retries,
        'difficulty_breakdown': {k: dict(v) for k, v in diff_breakdown.items()},
        'source_breakdown': {k: dict(v) for k, v in source_breakdown.items()}
    }

    # Write JSON summary
    json_path = results_dir / 'consensus_summary.json'
    # Save a clean version without giant problem lists for readability
    clean_json = {k: v for k, v in summary.items() if k != 'targeted_retries'}
    clean_json['targeted_retry_sample'] = {t: targeted_retries[t][:5] for t in teachers}
    json_path.write_text(json.dumps(clean_json, indent=2), encoding='utf-8')

    # Dynamic interpretations
    def get_interpretation(k, n):
        if k == n:
            return f"Universal / Easiest (All {n} models solved)"
        elif k == n - 1:
            return f"Near-Universal ({n-1} of {n} solved — 1 missed)"
        elif k >= 3:
            return f"High Consensus ({k} of {n} solved — {n-k} missed)"
        elif k == 2:
            return f"Challenging (Only 2 models solved)"
        elif k == 1:
            return f"Hard / Specialized (Only 1 model solved)"
        else:
            return f"Unsolved / Toughest (0 models solved)"

    # Write Markdown summary
    md_lines = [
        "# Cross-Model Consensus & Difficulty Analysis (Configuration B)",
        "",
        f"Evaluated across **{len(teachers)} teacher models** on the **{total_problems}-problem verified corpus**.",
        "",
        "## 1. Consensus Distribution (How many models solved each problem)",
        "",
        "| Models Solved | Problem Count | Percentage | Interpretation |",
        "|---|---|---|---|"
    ]

    for k in range(len(teachers), -1, -1):
        count = consensus_counts.get(k, 0)
        pct = (count / total_problems * 100) if total_problems else 0
        md_lines.append(f"| **{k} / {len(teachers)}** | **{count}** | **{pct:.1f}%** | {get_interpretation(k, len(teachers))} |")

    md_lines.extend([
        "",
        "## 2. Individual Model Performance",
        "",
        "| Teacher Model | Evaluated | Solved & Accepted | Acceptance Rate |",
        "|---|---|---|---|"
    ])

    for t in teachers:
        eval_count = model_evaluated.get(t, 0)
        solv_count = model_solves.get(t, 0)
        rate = (solv_count / eval_count * 100) if eval_count else 0
        md_lines.append(f"| `{t}` | {eval_count} / {total_problems} | **{solv_count}** | **{rate:.1f}%** |")

    target_str = ", ".join(str(k) for k in target_consensus)
    md_lines.extend([
        "",
        f"## 3. Targeted Retry Opportunities (Problems Solved by {target_str} Models)",
        "",
        f"A total of **{target_total} problems** have high multi-model consensus (solved by {target_str} models). "
        "These represent prime candidates for recovery via expanded feedback attempts (5 attempts), as they have been independently verified as solvable by multiple 7B teachers.",
        "",
        "| Model | Missed High-Consensus Problems | Potential Yield Boost |",
        "|---|---|---|"
    ])

    for t in teachers:
        missed_count = len(targeted_retries[t])
        current = model_solves.get(t, 0)
        max_potential = current + missed_count
        md_lines.append(f"| `{t}` | **{missed_count} problems** | {current} -> up to {max_potential} (+{missed_count/total_problems*100:.1f}%) |")

    md_lines.append("")

    md_path = results_dir / 'consensus_summary.md'
    md_path.write_text('\n'.join(md_lines), encoding='utf-8')
    logging.info("Saved consensus report to %s and %s", json_path, md_path)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Cross-Model Consensus Analysis")
    parser.add_argument('--config', default='config-v3.yaml')
    parser.add_argument('--teachers', nargs='+', help='Override active teacher list for consensus calculation')
    parser.add_argument('--consensus-levels', nargs='+', type=int, help='Consensus counts to target for retries (e.g. 3 4 5)')
    args = parser.parse_args()

    from .pipeline_v2 import load_config
    config = load_config(args.config)
    summary = analyze_consensus(config, teachers=args.teachers, target_consensus=args.consensus_levels)

    print("\n=======================================================")
    print("      CROSS-MODEL CONSENSUS ANALYSIS (CONFIG B)        ")
    print("=======================================================")
    for k in range(len(summary['teachers']), -1, -1):
        count = summary['consensus_counts'][k]
        pct = summary['consensus_percentages'][k]
        print(f"Solved by {k} models: {count:4d} ({pct:5.1f}%)")

    target_str = ", ".join(str(k) for k in summary['target_consensus'])
    print(f"\nTargeted 5-Attempt Retry Candidates (Solved by {target_str} models):")
    for t in summary['teachers']:
        missed = summary['targeted_retry_counts'][t]
        print(f"  {t:34s} missed {missed:3d} high-consensus problems")


if __name__ == '__main__':
    main()

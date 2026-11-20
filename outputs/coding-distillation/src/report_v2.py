import json
import logging
from pathlib import Path

from .common import jsonl, read_json, slug


def generate_report(config, model=None):
    results_dir = Path(config['results_dir']) / 'summaries'
    results_dir.mkdir(parents=True, exist_ok=True)
    teachers = [model] if model else config['teachers']

    corpus_file = Path(config['data_dir']) / 'problems' / 'verified_corpus.jsonl'
    problems = jsonl(corpus_file) if corpus_file.exists() else []

    full_report = {}

    for teacher in teachers:
        t_slug = slug(teacher)
        t_data = {'teacher': teacher, 'config_a': {}, 'config_b': {}}

        # Process Config A records
        dir_a = Path(config['data_dir']) / 'raw' / t_slug / 'config_a'
        if dir_a.exists():
            records_a = [read_json(f) for f in dir_a.glob('*/record.json')]
            accepted_a = sum(1 for r in records_a if r.get('status') == 'accepted')
            total_time_a = sum(r.get('generation_wall_seconds', 0) for r in records_a)
            t_data['config_a'] = {
                'total_problems': len(records_a),
                'accepted': accepted_a,
                'acceptance_rate': (accepted_a / len(records_a)) if records_a else 0,
                'total_generation_seconds': total_time_a,
                'avg_generation_seconds': (total_time_a / len(records_a)) if records_a else 0,
                'records': records_a
            }

        # Process Config B records
        dir_b = Path(config['data_dir']) / 'raw' / t_slug / 'config_b'
        if dir_b.exists():
            records_b = [read_json(f) for f in dir_b.glob('*/record.json')]
            accepted_b = sum(1 for r in records_b if r.get('status') == 'accepted')
            rejection_reasons = {}
            code_gen_seconds = 0
            val_seconds = 0
            exp_gen_seconds = 0
            max_cpu_seconds = 0
            max_peak_rss = 0
            infra_failures = 0

            for r in records_b:
                if 'attempts' in r and isinstance(r['attempts'], list):
                    code_gen_seconds += sum(a.get('generation_wall_seconds', 0) or 0 for a in r['attempts'])
                else:
                    code_gen_seconds += r.get('generation_wall_seconds', 0) or 0

                val = r.get('validation') or {}
                val_seconds += val.get('elapsed_seconds', 0) or 0
                exp_gen_seconds += r.get('explanation_wall_seconds', 0) or 0
                infra_failures += val.get('infrastructure_failures', 0) or 0

                for c in val.get('cases', []):
                    max_cpu_seconds = max(max_cpu_seconds, c.get('cpu_seconds', 0) or 0)
                    max_peak_rss = max(max_peak_rss, c.get('peak_rss_kb', 0) or 0)

                st = r.get('status', 'unknown')
                rejection_reasons[st] = rejection_reasons.get(st, 0) + 1

            t_data['config_b'] = {
                'total_problems': len(records_b),
                'accepted': accepted_b,
                'pass_rate': (accepted_b / len(records_b)) if records_b else 0,
                'rejection_reasons': rejection_reasons,
                'code_generation_seconds': code_gen_seconds,
                'validation_seconds': val_seconds,
                'explanation_generation_seconds': exp_gen_seconds,
                'max_cpu_seconds': max_cpu_seconds,
                'max_peak_rss_kb': max_peak_rss,
                'infrastructure_failures': infra_failures,
                'records': records_b
            }

        full_report[teacher] = t_data

    # Load existing JSON summary if present to preserve all teachers
    summary_file = results_dir / 'distillation_summary.json'
    existing_summary = {}
    if summary_file.exists():
        try:
            existing_summary = json.loads(summary_file.read_text(encoding='utf-8'))
        except Exception:
            pass

    for t, data in full_report.items():
        existing_summary[t] = {
            'config_a': {k: v for k, v in data['config_a'].items() if k != 'records'},
            'config_b': {k: v for k, v in data['config_b'].items() if k != 'records'}
        }
    summary_file.write_text(json.dumps(existing_summary, indent=2), encoding='utf-8')

    # Generate Markdown Report
    md_lines = [
        "# Coding Distillation Revised Experiment Report",
        "",
        "## Overview",
        "This report summarizes results for the two distillation configurations across FP16 teacher models:",
        "- **Configuration A (`reference_explanation_distillation`)**: Teachers explain verified reference solutions.",
        "- **Configuration B (`independent_solution_distillation`)**: Teachers independently solve problems, verified via fail-fast Docker validation before explanation.",
        "",
        "## Summary Results",
        ""
    ]

    for teacher, data in existing_summary.items():
        md_lines.append(f"### Teacher: `{teacher}`")
        ca = data.get('config_a') or {}
        cb = data.get('config_b') or {}

        if ca and ca.get('total_problems', 0) > 0:
            md_lines.append(f"#### Configuration A (Reference Explanation Distillation)")
            md_lines.append(f"- **Evaluated Problems**: {ca.get('total_problems', 0)}")
            md_lines.append(f"- **Accepted Explanations**: {ca.get('accepted', 0)} ({ca.get('acceptance_rate', 0):.1%})")
            md_lines.append(f"- **Total Generation Time**: {ca.get('total_generation_seconds', 0):.2f}s (avg: {ca.get('avg_generation_seconds', 0):.2f}s/problem)")
            md_lines.append("")

        if cb and cb.get('total_problems', 0) > 0:
            md_lines.append(f"#### Configuration B (Independent Solution Distillation)")
            md_lines.append(f"- **Evaluated Problems**: {cb.get('total_problems', 0)}")
            md_lines.append(f"- **Independently Solved & Accepted**: {cb.get('accepted', 0)} ({cb.get('pass_rate', 0):.1%})")
            md_lines.append(f"- **Rejection Breakdown**: {json.dumps(cb.get('rejection_reasons', {}))}")
            md_lines.append(f"- **Code Generation Time**: {cb.get('code_generation_seconds', 0):.2f}s")
            md_lines.append(f"- **Validation Time (Fail-Fast)**: {cb.get('validation_seconds', 0):.2f}s")
            md_lines.append(f"- **Explanation Generation Time**: {cb.get('explanation_generation_seconds', 0):.2f}s")
            md_lines.append(f"- **Max Case CPU Time**: {cb.get('max_cpu_seconds', 0):.3f}s")
            md_lines.append(f"- **Max Case Peak RSS**: {cb.get('max_peak_rss_kb', 0)} KB")
            md_lines.append(f"- **Docker Infrastructure Failures**: {cb.get('infrastructure_failures', 0)}")
            md_lines.append("")

    report_md_path = results_dir / 'distillation_summary.md'
    report_md_path.write_text('\n'.join(md_lines), encoding='utf-8')
    logging.info('Saved summary report to %s', report_md_path)
    print('\n'.join(md_lines))
    return existing_summary

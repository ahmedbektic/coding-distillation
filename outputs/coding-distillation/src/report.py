"""Produce a pilot-only report without mistaking pending common-set items for failures."""
import collections
import json
from pathlib import Path

from .common import atomic_json, atomic_text, jsonl, read_json, slug


def pilot_report(config, teacher):
    pilot_manifest = Path(config['results_dir']) / 'pilot-validated-problems.json'
    all_problems = jsonl(Path(config['data_dir']) / 'problems/selected.jsonl')
    if pilot_manifest.exists():
        indices = [x['selection_index'] for x in read_json(pilot_manifest)[:config['dataset']['pilot_count']]]
        problems = [all_problems[i] for i in indices]
    else:
        problems = all_problems[:config['dataset']['pilot_count']]
    rows = []
    for i, p in enumerate(problems):
        folder = Path(config['data_dir']) / 'raw' / slug(teacher) / p['id']
        result = read_json(folder/'result.json') if (folder/'result.json').exists() else {'status': 'pending'}
        v = read_json(folder/'validation.json') if (folder/'validation.json').exists() else {}
        attempts = [read_json(f) for f in sorted(folder.glob('attempt-*.json'))]
        rows.append({'index': i+1, 'problem_id': p['id'], 'difficulty': p['difficulty'], 'source': p['source'], 'url': p['url'],
                     'status': result['status'], 'reasons': dict(collections.Counter(c['status'] for c in v.get('cases', []))),
                     'support_issues': v.get('support_issues', []), 'out_of_spec_test_indices': v.get('out_of_spec_test_indices', []),
                     'tests_total': len(p['tests']['inputs']), 'tests_run': v.get('tests_run', 0),
                     'tests_passed': v.get('tests_passed', 0), 'infrastructure_failures': v.get('infrastructure_failures', 0),
                     'generation_seconds': sum(a.get('generation_wall_seconds', 0) for a in attempts),
                     'validation_seconds': v.get('elapsed_seconds', 0), 'attempts': len(attempts),
                     'prompt_tokens': next((a.get('response', {}).get('prompt_eval_count') for a in attempts if a.get('response')), None),
                     'output_tokens': next((a.get('response', {}).get('eval_count') for a in attempts if a.get('response')), None)})
    accepted = sum(r['status'] == 'accepted' for r in rows)
    summary = {'teacher': teacher, 'pilot_problems': len(rows), 'accepted': accepted, 'acceptance_rate': accepted/len(rows),
               'states': dict(collections.Counter(r['status'] for r in rows)),
               'difficulty_counts': dict(collections.Counter(r['difficulty'] for r in rows)),
               'test_status_counts': dict(sum((collections.Counter(r['reasons']) for r in rows), collections.Counter())),
               'generation_seconds': sum(r['generation_seconds'] for r in rows),
               'validation_seconds': sum(r['validation_seconds'] for r in rows),
               'infrastructure_failures': sum(r['infrastructure_failures'] for r in rows),
               'tests_total': sum(r['tests_total'] for r in rows), 'tests_run': sum(r['tests_run'] for r in rows),
               'rows': rows}
    root = Path(config['results_dir']) / 'summaries'
    atomic_json(root/(slug(teacher)+'-pilot.json'), summary)
    text = '# Fresh Qwen pilot\n\n' + json.dumps({k:v for k,v in summary.items() if k != 'rows'}, indent=2) + '\n\n'
    text += '| # | Problem ID | Difficulty | Result | Passed/run/total | Generation s | Validation s |\n|---|---|---|---|---|---|---|\n'
    for r in rows:
        text += f"| {r['index']} | {r['problem_id'][:10]} | {r['difficulty']} | {r['status']} | {r['tests_passed']}/{r['tests_run']}/{r['tests_total']} | {r['generation_seconds']:.2f} | {r['validation_seconds']:.2f} |\n"
    atomic_text(root/(slug(teacher)+'-pilot.md'), text)
    return summary


if __name__ == '__main__':
    from .pipeline import load_config
    s = pilot_report(load_config('config-v2.yaml'), 'qwen2.5:7b-instruct-fp16')
    print(json.dumps({k:v for k,v in s.items() if k != 'rows'}, indent=2))

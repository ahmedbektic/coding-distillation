from collections import Counter
from pathlib import Path

from .common import atomic_json, read_json, slug, write_jsonl
from .validate import extract


def export(config, problems, teachers):
    data = Path(config['data_dir'])
    summary = {}
    for teacher in teachers:
        rows, provenance, states = [], [], Counter()
        reasons, solved_difficulty = Counter(), Counter()
        measured = []
        for p in problems:
            folder = data / 'raw' / slug(teacher) / p['id']
            terminal = folder / 'result.json'
            if not terminal.exists():
                states['pending'] += 1
                continue
            result = read_json(terminal)
            states[result['status']] += 1
            if (folder / 'validation.json').exists():
                validation = read_json(folder / 'validation.json')
                for case in validation.get('cases', []):
                    reasons[case['status']] += 1
                    measured.append(case)
            if result['status'] != 'accepted':
                continue
            validation = read_json(folder / 'validation.json')
            attempt = read_json(folder / result['attempt_file'])
            code = attempt['code']
            if not validation['accepted'] or extract(code) != code or attempt['prompt'] != p['prompt']:
                raise ValueError('Export integrity check failed')
            rows.append({'prompt': p['prompt'], 'completion': code})
            solved_difficulty[p['difficulty']] += 1
            provenance.append({'line': len(rows), 'problem_id': p['id'], 'source': p['source'],
                               'url': p['url'], 'attempt': str((folder / result['attempt_file']).relative_to(data))})
        write_jsonl(data / 'verified' / (slug(teacher) + '.jsonl'), rows)
        write_jsonl(data / 'verified' / (slug(teacher) + '.provenance.jsonl'), provenance)
        summary[teacher] = {'common_problems': len(problems), 'solved': len(rows), 'states': dict(states),
                            'solved_by_difficulty': dict(solved_difficulty), 'test_status_counts': dict(reasons),
                            'max_case_cpu_seconds': max((x.get('cpu_seconds', 0) for x in measured), default=0),
                            'max_case_peak_rss_kb': max((x.get('peak_rss_kb', 0) for x in measured), default=0)}
    atomic_json(Path(config['results_dir']) / 'summaries' / f'latest-{len(problems)}.json', summary)
    return summary

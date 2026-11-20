"""Revalidate preserved generations and pinned dataset references, never on host."""
import collections
from pathlib import Path

from .common import atomic_json, digest, jsonl, read_json
from .pipeline import load_config
from .validate import DockerValidator


def run():
    config = load_config('config-v2.yaml')
    root = Path(config['results_dir']) / 'reference-validation-confirmed'
    root.mkdir(parents=True, exist_ok=True)
    validator = DockerValidator(config['validation'])
    problems = jsonl(Path(config['data_dir']) / 'problems/selected.jsonl')
    indices = [0, 3, 6, 7, 9, 10, 12, 15, 18]
    indices += [i for i, p in enumerate(problems) if p['source'] == 'leetcode'][:1]
    summary = []
    for i in indices:
        p = problems[i]
        references = read_json(Path(config['results_dir']) / 'reference-sources' / (p['id']+'.json'))['references']
        for ref in references:
            file = root / f'{i:04d}-ref-{ref["reference_index"]}.json'
            record = {'problem_id': p['id'], 'selection_index': i, 'reference': ref,
                      'source': p['source'], 'format': 'callable' if p['tests']['fn_name'] else 'stdio'}
            if file.exists():
                record = read_json(file)
                if record['validation']['image_id'] != validator.image:
                    raise RuntimeError('Reference checks already exist for a different image; choose new result directory')
            else:
                record['validation'] = validator.validate(ref['code'], p)
                atomic_json(file, record)
            v = record['validation']
            summary.append({'index': i, 'problem_id': p['id'], 'reference_index': ref['reference_index'],
                            'accepted': v['accepted'], 'status': v['status'], 'cases': v['tests_run'],
                            'passed': v['tests_passed'], 'infrastructure_failures': v['infrastructure_failures'],
                            'elapsed': v['elapsed_seconds'], 'categories': dict(collections.Counter(c['status'] for c in v['cases']))})
            print(summary[-1], flush=True)
            if v['accepted']:
                break
    # Known-negative controls use the same real callable problem and exact resource policy.
    p = problems[6]
    fn = p['tests']['fn_name']
    controls = {'wrong_answer': f'def {fn}(*args): return -123456789',
                'inefficient': f'def {fn}(*args):\n while True: pass',
                'syntax_error': 'def broken(:',
                'signature': f'def {fn}(): return 0'}
    for name, code in controls.items():
        file = root / ('negative-'+name+'.json')
        v = read_json(file) if file.exists() else validator.validate(code, p)
        atomic_json(file, v)
        assert not v['accepted'], name
        print('CONTROL', name, v['status'], dict(collections.Counter(c['status'] for c in v['cases'])), flush=True)
    for p in problems[:2]:
        original = Path('data/raw/qwen2.5_7b-instruct-fp16') / p['id']
        generation = read_json(original / 'attempt-01.json')
        v = validator.validate(generation['code'], p)
        atomic_json(root / ('old-generation-' + p['id'] + '.json'), v)
        print('OLD GENERATION', p['id'][:10], dict(collections.Counter(c['status'] for c in v['cases'])), flush=True)
    atomic_json(root / 'summary.json', summary)


if __name__ == '__main__':
    run()

"""Read-only audit of the frozen selection and pinned reference solutions."""
import ast
import collections
import json
from pathlib import Path

from .checkers import checker_name, support_issues, invalid_test_indices
from .common import atomic_json, digest, jsonl, read_json, write_jsonl
from .dataset import arrow_rows, keys, problem


def audit(config):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    data = Path(config['data_dir'])
    out = Path(config['results_dir'])
    manifest = read_json(data / 'problems/manifest.json')
    selected = jsonl(data / 'problems/selected.jsonl')
    assert digest(selected) == manifest['selection_hash']
    snapshot = Path(config['dataset']['cache_dir']) / 'datasets--BAAI--TACO/snapshots' / manifest['revision']
    eval_keys = set().union(*(keys(r) for r in arrow_rows(snapshot / 'test/data-00000-of-00001.arrow')))
    seen = set()
    indexed = {p['source_index']: p for p in selected}
    rows = []
    for p in selected:
        k = keys(p)
        rows.append({'id': p['id'], 'source_index': p['source_index'], 'source': p['source'],
                     'difficulty': p['difficulty'], 'url': p['url'], 'format': 'callable' if p['tests']['fn_name'] else 'stdio',
                     'split': p['split'], 'test_count': len(p['tests']['inputs']),
                     'duplicate': bool(k & seen), 'evaluation_overlap': bool(k & eval_keys),
                     'support_issues': support_issues(p), 'checker': checker_name(p),
                     'out_of_spec_test_indices': invalid_test_indices(p),
                     'max_expected_bytes': max(len(json.dumps(v).encode()) if p['tests']['fn_name'] else len(v.encode()) for v in p['tests']['outputs'])})
        seen.update(k)
    records = {r['id']: r for r in rows}
    position = 0
    for filename in sorted(snapshot.glob('train/*.arrow')):
        print('Auditing', filename.name, flush=True)
        with pa.memory_map(str(filename), 'r') as stream:
            reader = ipc.open_stream(stream)
            for batch in reader:
                for idx in range(position, position+batch.num_rows):
                    if idx not in indexed:
                        continue
                    p = indexed[idx]
                    original = batch.slice(idx-position, 1).to_pylist()[0]
                    row = records[p['id']]
                    reconstructed = problem(original, idx, config['dataset']['min_tests'], config['dataset']['max_prompt_chars'])
                    row['matches_pinned_train_row'] = reconstructed == p
                    try:
                        candidates = json.loads(original['solutions'])
                        if not isinstance(candidates, list):
                            candidates = []
                    except (ValueError, TypeError):
                        candidates = []
                    usable = []
                    for number, code in enumerate(candidates):
                        if not isinstance(code, str):
                            continue
                        try:
                            ast.parse(code)
                        except (SyntaxError, ValueError):
                            continue
                        usable.append({'reference_index': number, 'code': code, 'code_hash': digest(code)})
                        if len(usable) == 3:
                            break
                    row['reference_count'] = len(candidates)
                    row['python3_parseable_references_saved'] = len(usable)
                    if not usable:
                        row['support_issues'].append('no_parseable_python3_reference')
                    atomic_json(out / 'reference-sources' / (p['id'] + '.json'),
                                {'problem_id': p['id'], 'revision': manifest['revision'], 'source_index': idx, 'references': usable})
                position += batch.num_rows
    summary = {'selection_hash': digest(selected), 'revision': manifest['revision'], 'selected': len(rows),
               'train_rows_scanned': position, 'all_match_pinned_train': all(r.get('matches_pinned_train_row') for r in rows),
               'duplicate_count': sum(r['duplicate'] for r in rows), 'evaluation_overlap_count': sum(r['evaluation_overlap'] for r in rows),
               'issues': dict(collections.Counter(i for r in rows for i in r['support_issues'])),
               'known_out_of_spec_problems': sum(bool(r['out_of_spec_test_indices']) for r in rows),
               'unflagged': sum(not r['support_issues'] for r in rows),
               'max_expected_bytes': max(r['max_expected_bytes'] for r in rows),
               'test_cases': sum(r['test_count'] for r in rows)}
    write_jsonl(out / 'dataset-support.jsonl', rows)
    atomic_json(out / 'dataset-support-summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    from .pipeline import load_config
    audit(load_config('config-v2.yaml'))

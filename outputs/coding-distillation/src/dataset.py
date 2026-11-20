"""Load pinned Arrow data without executing the dataset's legacy Python script."""
import ast
import collections
import json
import random
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit

from .common import atomic_json, digest, jsonl, read_json, write_jsonl


def normalized(text):
    return re.sub(r'\W+', '', unicodedata.normalize('NFKC', text).casefold())


def keys(row):
    text = row.get('question') or row.get('prompt') or ''
    result = {'q:' + digest(normalized(text))} if text.strip() else set()
    url = row.get('url') or ''
    if url:
        p = urlsplit(url)
        result.add('url:' + p.netloc.lower() + p.path.rstrip('/'))
    return result


def labels(value):
    if isinstance(value, list):
        return sorted(str(x) for x in value) or ['unknown']
    if not value:
        return ['unknown']
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return sorted(str(x) for x in parsed) or ['unknown']
    except (ValueError, TypeError):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return sorted(str(x) for x in parsed) or ['unknown']
        except (ValueError, SyntaxError, TypeError):
            pass
    return [str(value)]


def problem(row, index, min_tests=2, max_prompt_chars=6500):
    question = row.get('question', '').strip()
    if re.search(r'\b(?:this is an interactive problem|interaction\s*\n)', question, re.I):
        raise ValueError('interactive problem')
    if '<image>' in question or re.search(r'!\[[^\]]*\]\(', question):
        raise ValueError('image-dependent statement')
    io = row.get('input_output')
    io = json.loads(io) if isinstance(io, str) else io
    if not isinstance(io, dict):
        raise ValueError('missing tests')
    ins, outs = io.get('inputs'), io.get('outputs')
    if not isinstance(ins, list) or not isinstance(outs, list) or len(ins) != len(outs) or len(ins) < min_tests:
        raise ValueError('invalid/insufficient tests')
    fn = io.get('fn_name') or None
    if fn and (not isinstance(fn, str) or not fn.isidentifier()):
        raise ValueError('unsupported function name')
    if fn:
        if not all(isinstance(x, list) for x in ins):
            raise ValueError('call inputs must be argument lists')
        source = (row.get('source') or '').lower()
        if source == 'codewars':
            if not all(isinstance(x, list) and len(x) == 1 for x in outs):
                raise ValueError('unsupported Codewars output envelope')
            # TACO Codewars uses one outer result envelope; preserve actual list answers inside it.
            outs = [x[0] for x in outs]
        if source == 'geeksforgeeks':
            raise ValueError('GFG callable uses driver-formatted outputs; adapter not available')
    elif not all(isinstance(x, str) for x in ins + outs):
        raise ValueError('stdio inputs and expected outputs must be strings')
    if not question:
        raise ValueError('empty question')
    starter = row.get('starter_code') or ''
    entry = (f'Implement the callable {fn}, preserving the supplied signature and Solution class if present. '
             'Return the answer; do not read stdin.' if fn else
             'Write a complete program that reads stdin and writes only the required answer to stdout.')
    prompt = ('Solve the following algorithm problem in Python 3. Use a correct, efficient algorithm '
              'and handle boundary cases within the stated constraints. Return only complete Python code, '
              'without Markdown fences or explanations. Use only the Python standard library.\n\n'
              + entry + '\n\n' + question)
    if starter.strip():
        prompt += '\n\nStarter code (preserve its public interface):\n' + starter.strip()
    if len(prompt) > max_prompt_chars:
        raise ValueError('prompt character budget exceeded')
    return {'id': digest(normalized(question)), 'source_index': index, 'split': 'train',
            'question': question, 'prompt': prompt, 'tests': {'inputs': ins, 'outputs': outs, 'fn_name': fn},
            'difficulty': row.get('difficulty') or 'unknown', 'tags': labels(row.get('tags')),
            'skills': labels(row.get('skill_types')), 'source': row.get('source'), 'url': row.get('url'),
            'starter_code': starter, 'test_adapter': 'codewars-singleton-envelope' if fn and (row.get('source') or '').lower() == 'codewars' else 'direct',
            'reference': {k: row.get(k) for k in
                ('time_limit', 'memory_limit', 'Expected Time Complexity', 'Expected Auxiliary Space')}}


def select(train, evaluation, count, seed, min_tests=2, max_prompt_chars=6500):
    blocked = set().union(*(keys(r) for r in evaluation)) if evaluation else set()
    seen = set()
    groups = collections.defaultdict(list)
    audit = collections.Counter()
    for i, row in enumerate(train):
        identifiers = keys(row)
        if identifiers & blocked:
            audit['evaluation_overlap'] += 1
            continue
        if identifiers & seen:
            audit['duplicate'] += 1
            continue
        seen.update(identifiers)
        try:
            p = problem(row, i, min_tests, max_prompt_chars)
        except (ValueError, TypeError, KeyError) as e:
            audit['excluded: ' + str(e)] += 1
            continue
        groups[(p['difficulty'], p['skills'][0], p['tags'][0])].append(p)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    # Rotate difficulty first, then algorithm strata within each difficulty.
    difficulties = sorted({k[0] for k in groups})
    strata = {d: [k for k in sorted(groups) if k[0] == d] for d in difficulties}
    for values in strata.values():
        rng.shuffle(values)
    result = []
    while len(result) < count:
        added = False
        for d in difficulties:
            active = strata[d]
            while active and not groups[active[0]]:
                active.pop(0)
            if active and len(result) < count:
                key = active.pop(0)
                result.append(groups[key].pop())
                active.append(key)
                added = True
        if not added:
            raise ValueError(f'Only {len(result)} eligible unique problems; requested {count}')
    audit['selected'] = len(result)
    return result, dict(audit)


def arrow_rows(path):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.memory_map(str(path), 'r') as source:
        try:
            reader = ipc.open_stream(source)
            batches = reader
        except pa.ArrowInvalid:
            source.seek(0)
            reader = ipc.open_file(source)
            batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
        for batch in batches:
            # Reference solutions are never used or copied to prompts.
            columns = [n for n in batch.schema.names if n != 'solutions']
            yield from batch.select(columns).to_pylist()


def prepare(config, fixture=None):
    dst = Path(config['data_dir']) / 'problems'
    selection_config = {'dataset': config['dataset'], 'seed': config['seed'],
                        'fixture_sha': digest(jsonl(fixture)) if fixture else None}
    extras = [row for f in config['dataset']['exclusion_files'] for row in jsonl(f)]
    selection_config['exclusion_hash'] = digest(extras)
    manifest_path = dst / 'manifest.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest['selection_config'] != selection_config:
            raise ValueError('Selection configuration changed. Use a new data_dir.')
        rows = jsonl(dst / 'selected.jsonl')
        if digest(rows) != manifest['selection_hash']:
            raise ValueError('Selection file hash mismatch')
        return manifest
    if fixture:
        rows = jsonl(fixture)
        train = [r for r in rows if r.get('split', 'train') == 'train']
        evaluation = [r for r in rows if r.get('split') == 'test'] + extras
        revision, files = 'local-fixture', []
    else:
        from huggingface_hub import HfApi, hf_hub_download
        ds = config['dataset']
        source_path = dst / 'source.json'
        if source_path.exists():
            source_lock = read_json(source_path)
            if source_lock['repository'] != ds['repository'] or source_lock['requested_revision'] != ds['revision']:
                raise ValueError('Dataset source changed; use a new data_dir')
            revision = source_lock['revision']
        else:
            revision = HfApi().dataset_info(ds['repository'], revision=ds['revision']).sha
            atomic_json(source_path, {'repository': ds['repository'], 'requested_revision': ds['revision'], 'revision': revision})
        info = HfApi().dataset_info(ds['repository'], revision=revision)
        files = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith('.arrow')
                       and s.rfilename.startswith(('train/', 'test/')))
        if not any(f.startswith('train/') for f in files) or not any(f.startswith('test/') for f in files):
            raise ValueError('Expected TACO train/test Arrow files not found')
        def split_rows(split):
            for name in files:
                if name.startswith(split + '/'):
                    print('Loading', name, flush=True)
                    path = hf_hub_download(ds['repository'], name, repo_type='dataset', revision=revision, cache_dir=ds['cache_dir'])
                    yield from arrow_rows(path)
        evaluation = list(split_rows('test')) + extras
        train = split_rows('train')
    selected, audit = select(train, evaluation, config['dataset']['count'], config['seed'],
                             config['dataset']['min_tests'], config['dataset']['max_prompt_chars'])
    manifest = {'repository': config['dataset']['repository'], 'revision': revision, 'files': files,
                'selection_config': selection_config, 'selection_hash': digest(selected), 'audit': audit,
                'difficulty_counts': dict(collections.Counter(p['difficulty'] for p in selected)),
                'tag_counts': dict(collections.Counter(t for p in selected for t in p['tags'])),
                'evaluation_records_checked': len(evaluation), 'fixture': bool(fixture)}
    write_jsonl(dst / 'selected.jsonl', selected)
    atomic_json(manifest_path, manifest)
    return manifest

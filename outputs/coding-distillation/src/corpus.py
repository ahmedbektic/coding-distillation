import ast
import json
import logging
import re
from pathlib import Path
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from .common import atomic_json, digest, jsonl, write_jsonl
from .dataset import normalized, problem as taco_problem
from .validate import DockerValidator


def parse_mbpp_tests(test_list):
    inputs = []
    outputs = []
    fn_name = None
    for test_str in test_list:
        tree = ast.parse(test_str.strip()).body[0]
        if not isinstance(tree, ast.Assert):
            continue
        comp = tree.test
        if not isinstance(comp, ast.Compare) or not comp.comparators:
            continue
        if not isinstance(comp.left, ast.Call) or not hasattr(comp.left.func, 'id'):
            continue
        fn = comp.left.func.id
        if fn_name is None:
            fn_name = fn
        elif fn != fn_name:
            continue
        args = [ast.literal_eval(a) for a in comp.left.args]
        expected = ast.literal_eval(comp.comparators[0])
        inputs.append(args)
        outputs.append(expected)
    if not fn_name or len(inputs) < 2:
        raise ValueError('Could not extract at least 2 valid function test cases')
    return fn_name, inputs, outputs


def mbpp_to_problem(row):
    fn_name, inputs, outputs = parse_mbpp_tests(row['test_list'])
    question = row['text'].strip()
    task_id = row['task_id']
    prompt = (
        "Solve the following algorithm problem in Python 3. Use a correct, efficient algorithm "
        "and handle boundary cases within the stated constraints. Return only complete Python code, "
        "without Markdown fences or explanations. Use only the Python standard library.\n\n"
        f"Implement the callable {fn_name}, preserving the supplied signature. "
        "Return the answer; do not read stdin.\n\n"
        f"{question}"
    )
    # Infer algorithm category
    lower_q = question.lower()
    tags = []
    for cat in ('array', 'list', 'string', 'math', 'dict', 'dictionary', 'set', 'sort', 'search', 'recursion', 'tree', 'graph'):
        if cat in lower_q:
            tags.append(cat)
    if not tags:
        tags = ['general']

    return {
        'id': digest(f"mbpp_{task_id}_{normalized(question)}"),
        'source': 'mbpp',
        'source_id': str(task_id),
        'difficulty': 'EASY',
        'tags': tags,
        'format': 'callable',
        'fn_name': fn_name,
        'question': question,
        'prompt': prompt,
        'starter_code': '',
        'tests': {'inputs': inputs, 'outputs': outputs, 'fn_name': fn_name},
        'provenance': {
            'dataset': 'google-research-datasets/mbpp',
            'split': 'train',
            'license': 'Apache-2.0',
            'url': f'https://github.com/google-research/google-research/tree/master/mbpp'
        }
    }


def load_mbpp_candidates(cache_dir='../../work/hf-cache'):
    parquet_path = hf_hub_download(
        'google-research-datasets/mbpp',
        'full/train-00000-of-00001.parquet',
        repo_type='dataset',
        cache_dir=cache_dir
    )
    table = pq.read_table(parquet_path)
    candidates = []
    for row in table.to_pylist():
        try:
            prob = mbpp_to_problem(row)
            ref_code = row['code'].replace('\r\n', '\n').strip() + '\n'
            candidates.append((prob, ref_code))
        except Exception:
            continue
    return candidates


def load_taco_codewars_candidates(cache_dir='../../work/hf-cache', max_candidates=2500):
    candidates = []
    shards = list(Path(cache_dir).glob('**/train/*.arrow'))
    if not shards:
        return candidates

    seen_prompts = set()
    idx = 0
    for shard in shards:
        try:
            reader = ipc.open_stream(shard.read_bytes())
        except Exception:
            continue
        for batch in reader:
            rows = batch.to_pylist()
            for row in rows:
                source = (row.get('source') or '').lower()
                # Strictly exclude Codeforces and LeetCode
                if 'codeforces' in source or 'leetcode' in source:
                    continue
                # Priority 2: Codewars
                if source != 'codewars':
                    continue

                sols = row.get('solutions')
                if not sols:
                    continue
                if isinstance(sols, str):
                    try:
                        sols = json.loads(sols)
                    except Exception:
                        continue
                if not isinstance(sols, list) or not sols or not isinstance(sols[0], str) or not sols[0].strip():
                    continue

                q_norm = normalized(row.get('question', ''))
                if q_norm in seen_prompts:
                    continue
                seen_prompts.add(q_norm)

                try:
                    p = taco_problem(row, idx)
                except Exception:
                    continue

                p['source'] = 'codewars'
                p['fn_name'] = p['tests'].get('fn_name')
                p['provenance'] = {
                    'dataset': 'BAAI/TACO',
                    'split': 'train',
                    'source': 'codewars',
                    'license': 'MIT / CC BY-SA 4.0 / Apache-2.0',
                    'url': row.get('url', '')
                }
                candidates.append((p, sols[0].replace('\r\n', '\n').strip() + '\n'))
                idx += 1
                if len(candidates) >= max_candidates:
                    return candidates
    return candidates


def build_verified_corpus(config, target_count=1000, validator=None):
    if validator is None:
        validator = DockerValidator(config['validation'])

    dst = Path(config['data_dir']) / 'problems'
    dst.mkdir(parents=True, exist_ok=True)
    corpus_file = dst / 'verified_corpus.jsonl'
    manifest_file = dst / 'corpus_manifest.json'

    verified_corpus = []
    seen_ids = set()
    audit = {'evaluated': 0, 'accepted': 0, 'rejected': 0, 'by_source': {}}

    if corpus_file.exists() and manifest_file.exists():
        existing = jsonl(corpus_file)
        if len(existing) >= target_count:
            logging.info('Reusing existing verified reference corpus from %s (%d problems)', corpus_file, len(existing))
            return existing, json.loads(manifest_file.read_text(encoding='utf-8'))
        else:
            logging.info('Found existing %d problems in %s; expanding to target %d...', len(existing), corpus_file, target_count)
            verified_corpus = existing
            seen_ids = {p['id'] for p in existing}
            audit['accepted'] = len(existing)
            audit['evaluated'] = len(existing)
            for p in existing:
                src = p.get('source', 'unknown')
                audit['by_source'].setdefault(src, {'evaluated': 0, 'accepted': 0})
                audit['by_source'][src]['accepted'] += 1
                audit['by_source'][src]['evaluated'] += 1

    logging.info('Building verified Python reference corpus (target: %d)...', target_count)
    cache_dir = config['dataset'].get('cache_dir', '../../work/hf-cache')

    # Step 1: MBPP train candidates
    logging.info('Loading MBPP train candidates...')
    mbpp_cands = load_mbpp_candidates(cache_dir)
    logging.info('Loaded %d MBPP train candidates.', len(mbpp_cands))

    # Step 2: TACO Codewars candidates
    logging.info('Loading TACO Codewars candidates...')
    codewars_cands = load_taco_codewars_candidates(cache_dir, max_candidates=target_count * 2)
    logging.info('Loaded %d Codewars candidates.', len(codewars_cands))

    all_candidates = mbpp_cands + codewars_cands

    for prob, ref_code in all_candidates:
        if len(verified_corpus) >= target_count:
            break
        if prob['id'] in seen_ids:
            continue

        audit['evaluated'] += 1
        res = validator.validate(ref_code, prob)
        src = prob['source']
        audit['by_source'].setdefault(src, {'evaluated': 0, 'accepted': 0})
        audit['by_source'][src]['evaluated'] += 1

        if res['accepted']:
            prob['reference_solution'] = ref_code
            prob['reference_validation'] = {
                'status': 'accepted',
                'tests_total': res['tests_total'],
                'tests_passed': res['tests_passed'],
                'elapsed_seconds': res['elapsed_seconds']
            }
            verified_corpus.append(prob)
            seen_ids.add(prob['id'])
            audit['accepted'] += 1
            audit['by_source'][src]['accepted'] += 1
            if len(verified_corpus) % 100 == 0:
                logging.info('Verified corpus progress: %d/%d accepted.', len(verified_corpus), target_count)
        else:
            audit['rejected'] += 1

    logging.info('Verified corpus complete: %d problems accepted out of %d evaluated.',
                 len(verified_corpus), audit['evaluated'])

    manifest = {
        'target_count': target_count,
        'actual_count': len(verified_corpus),
        'sources_included': list(audit['by_source'].keys()),
        'sources_excluded': ['codeforces', 'leetcode'],
        'audit': audit,
        'corpus_hash': digest(verified_corpus)
    }

    write_jsonl(corpus_file, verified_corpus)
    atomic_json(manifest_file, manifest)
    return verified_corpus, manifest

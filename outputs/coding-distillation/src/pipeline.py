"""CLI: python -m src.pipeline --help"""
import argparse
import importlib.metadata
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml
from filelock import FileLock

from .common import atomic_json, digest, jsonl, read_json, slug
from .dataset import prepare
from .export import export
from .generate import Ollama
from .validate import DockerValidator, extract


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    for key in ('data_dir', 'results_dir'):
        config[key] = str((path.parent / config[key]).resolve())
    config['dataset']['exclusion_files'] = [str((path.parent / p).resolve()) for p in config['dataset']['exclusion_files']]
    config['dataset']['cache_dir'] = str((path.parent / config['dataset']['cache_dir']).resolve())
    return config


def setup_logging(config):
    logs = Path(config['results_dir']) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(logs / 'pipeline.log', maxBytes=5_000_000, backupCount=5, encoding='utf-8')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), handler], force=True)


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Experiment changed: {path}. Use a new data_dir/results_dir.')
    else:
        atomic_json(path, value)


def solve(config, client, validator, teacher, p, metadata):
    folder = Path(config['data_dir']) / 'raw' / slug(teacher) / p['id']
    folder.mkdir(parents=True, exist_ok=True)
    terminal = folder / 'result.json'
    if terminal.exists():
        term = read_json(terminal)
        if not term.get('transient') and term.get('status') != 'docker_infrastructure_failure':
            return term['status']
    attempts = config['ollama']['transport_attempts']
    chosen = None
    for number in range(1, attempts + 1):
        path = folder / f'attempt-{number:02d}.json'
        if path.exists():
            record = read_json(path)
            if record.get('response') is not None:
                chosen = (path, record)
                break
            # Interrupted calls count against the retry budget; never overwrite them.
            continue
        record = {'model': teacher, 'problem_id': p['id'], 'prompt': p['prompt'], 'started_at': timestamp(),
                  'generation_parameters': config['ollama'], 'metadata_hash': digest(metadata),
                  'attempt': number, 'state': 'started', 'response': None, 'code': None}
        atomic_json(path, record)
        start = time.monotonic()
        try:
            response = client.generate(teacher, p['prompt'])
            record.update(response=response, state='generated', generation_wall_seconds=time.monotonic()-start)
            # Save raw output before any extraction or execution, so validation can resume without inference.
            atomic_json(path, record)
            chosen = (path, record)
            break
        except Exception as e:
            record.update(state='transport_error', error=f'{type(e).__name__}: {e}',
                          generation_wall_seconds=time.monotonic()-start)
            atomic_json(path, record)
            logging.warning('%s %s attempt %d: %s', teacher, p['id'][:10], number, e)
            if number < attempts:
                time.sleep(min(2 ** number, 8))
    if chosen is None:
        atomic_json(terminal, {'status': 'generation_failed', 'finished_at': timestamp()})
        return 'generation_failed'
    path, record = chosen
    try:
        response = record['response']
        if not response.get('done') or response.get('done_reason') == 'length':
            raise ValueError('incomplete or output-truncated generation')
        # Ollama exposes evaluated prompt tokens but no exact portable preflight tokenizer.
        if not isinstance(response.get('prompt_eval_count'), int):
            raise ValueError('missing prompt token accounting')
        options = config['ollama']['options']
        if response['prompt_eval_count'] + options['num_predict'] + 256 > options['num_ctx']:
            raise ValueError('context budget suspect')
        if record.get('code') is None:
            record['code'] = extract(response['message']['content'])
            record['state'] = 'extracted'
            atomic_json(path, record)
    except Exception as e:
        category = 'syntax_error' if isinstance(e, SyntaxError) else 'extraction_failed'
        record.update(state=category, error=f'{type(e).__name__}: {e}')
        atomic_json(path, record)
        atomic_json(terminal, {'status': category, 'attempt_file': path.name, 'finished_at': timestamp()})
        return category
    validation_path = folder / 'validation.json'
    if validation_path.exists():
        validation = read_json(validation_path)
        if validation.get('status') == 'docker_infrastructure_failure':
            validation = validator.validate(record['code'], p)
            atomic_json(validation_path, validation)
    else:
        validation = validator.validate(record['code'], p)
        atomic_json(validation_path, validation)
    status = 'accepted' if validation['accepted'] else validation.get('status', 'rejected')
    if status == 'docker_infrastructure_failure':
        atomic_json(terminal, {'status': status, 'attempt_file': path.name, 'transient': True, 'finished_at': timestamp()})
    else:
        atomic_json(terminal, {'status': status, 'attempt_file': path.name, 'finished_at': timestamp()})
    return status


def run(config, mode, model=None):
    manifest = read_json(Path(config['data_dir']) / 'problems' / 'manifest.json')
    selection = manifest['selection_config']
    if selection['dataset'] != config['dataset'] or selection['seed'] != config['seed']:
        raise ValueError('Prepared selection configuration differs from this run')
    external = [row for f in config['dataset']['exclusion_files'] for row in jsonl(f)]
    if digest(external) != selection['exclusion_hash']:
        raise ValueError('Evaluation exclusions changed after selection')
    all_problems = jsonl(Path(config['data_dir']) / 'problems' / 'selected.jsonl')
    if digest(all_problems) != manifest['selection_hash']:
        raise ValueError('Frozen selection was modified')
    if mode == 'full' and manifest['fixture']:
        raise ValueError('Full research runs cannot use fixture data')
    if mode == 'full':
        from .checkers import support_issues, invalid_test_indices
        flagged = {p['id']: support_issues(p) for p in all_problems if support_issues(p) or invalid_test_indices(p)}
        if flagged:
            raise ValueError(f'Full run blocked: {len(flagged)} frozen problems require judge/format review. '
                             'See dataset-support.jsonl. Do not replace them silently.')
    count = {'pilot': config['dataset']['pilot_count'], 'integration': 2, 'full': config['dataset']['count']}[mode]
    if count > len(all_problems):
        raise ValueError('Not enough prepared problems')
    if mode == 'pilot':
        pilot_manifest = Path(config['results_dir']) / 'pilot-validated-problems.json'
        if pilot_manifest.exists():
            pilot_records = read_json(pilot_manifest)
            pilot_indices = [r['selection_index'] for r in pilot_records[:count]]
            problems = [all_problems[i] for i in pilot_indices]
            logging.info('Pilot mode: using %d independently validated problems from %s', len(problems), pilot_manifest.name)
        else:
            problems = all_problems[:count]
    else:
        problems = all_problems[:count]
    teachers = [model] if model else config['teachers']
    if any(t not in config['teachers'] for t in teachers):
        raise ValueError('Teacher must be one of the configured FP16 teachers')
    validator = DockerValidator(config['validation'])  # fail closed before inference
    client = Ollama(config['ollama'])
    source_hash = digest({str(p.relative_to(Path(__file__).parent)): p.read_text(encoding='utf-8')
                          for p in sorted(Path(__file__).parent.glob('*.py'))})
    experiment = {'config': config, 'selection_hash': manifest['selection_hash'],
                  'image_id': validator.image, 'python': platform.python_version(), 'source_hash': source_hash,
                  'packages': {name: importlib.metadata.version(name) for name in ('PyYAML', 'huggingface-hub', 'pyarrow', 'filelock')}}
    freeze(Path(config['data_dir']) / 'experiment.json', experiment)
    for teacher in teachers:
        try:
            metadata = client.metadata(teacher)
            freeze(Path(config['data_dir']) / 'raw' / slug(teacher) / 'metadata.json', metadata)
            client.unload_all()
        except Exception:
            logging.exception('Teacher preflight failed: %s', teacher)
            atomic_json(Path(config['results_dir']) / 'summaries' / (slug(teacher) + '-preflight-error.json'),
                        {'at': timestamp(), 'error': str(sys.exc_info()[1])})
            continue
        try:
            for i, p in enumerate(problems, 1):
                try:
                    status = solve(config, client, validator, teacher, p, metadata)
                except Exception as e:
                    logging.exception('Problem failed: %s', p['id'])
                    # Persist infrastructure failure separately; resume revalidates saved generation.
                    atomic_json(Path(config['data_dir']) / 'raw' / slug(teacher) / p['id'] / 'infrastructure-error.json',
                                {'error': str(e), 'at': timestamp()})
                    status = 'infrastructure_error'
                logging.info('%s [%d/%d] %s %s', teacher, i, count, p['id'][:10], status)
        finally:
            try:
                client.unload(teacher)
            except Exception:
                logging.exception('Failed to unload %s; stopping to avoid overlapping teachers', teacher)
                raise
            export(config, all_problems, config['teachers'])
    summary = export(config, all_problems, config['teachers'])
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Freeze common training selection; downloads TACO shards')
    prep.add_argument('--fixture')
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--mode', choices=['pilot', 'full', 'integration'], default='pilot')
    run_parser.add_argument('--model')
    sub.add_parser('doctor')
    sub.add_parser('export')
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config)
    Path(config['data_dir']).mkdir(parents=True, exist_ok=True)
    with FileLock(str(Path(config['data_dir']) / '.pipeline.lock'), timeout=0):
        if args.command == 'prepare':
            print(json.dumps(prepare(config, args.fixture), indent=2))
        elif args.command == 'run':
            run(config, args.mode, args.model)
        elif args.command == 'export':
            print(json.dumps(export(config, jsonl(Path(config['data_dir']) / 'problems' / 'selected.jsonl'), config['teachers']), indent=2))
        else:
            client = Ollama(config['ollama'])
            print('Ollama:', client.api('/api/version'))
            for t in config['teachers']:
                meta = client.metadata(t)
                print(t, meta['tag']['digest'], meta['tag']['details']['quantization_level'])
            print('Sandbox image:', DockerValidator(config['validation']).image)


if __name__ == '__main__':
    main()

"""Revised Coding Distillation Pipeline supporting Configuration A and Configuration B."""
import argparse
import importlib.metadata
import json
import logging
import platform
import sys
import time
from pathlib import Path

from filelock import FileLock

from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from .common import atomic_json, digest, jsonl, read_json, slug, write_jsonl
from .corpus import build_verified_corpus
from .explanation import build_explanation_prompt, verify_explanation
from .generate import Ollama
from .repair import (
    build_failure_feedback,
    build_initial_prompt,
    build_repair_prompt,
    extract_function_signature,
)
from .validate import DockerValidator, extract


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def setup_logging(config):
    logs = Path(config['results_dir']) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(logs / 'pipeline.log', maxBytes=5_000_000, backupCount=5, encoding='utf-8')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), handler], force=True)


def load_config(path='config-v3.yaml'):
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def solve_config_a(config, client, teacher, problem, metadata):
    """
    Configuration A: Reference-Solution Explanation Distillation.
    Teacher explains an existing, verified reference solution.
    """
    record_dir = Path(config['data_dir']) / 'raw' / slug(teacher) / 'config_a' / problem['id']
    record_file = record_dir / 'record.json'
    if record_file.exists():
        existing = read_json(record_file)
        if not existing.get('transient'):
            return existing.get('status', 'accepted')

    record_dir.mkdir(parents=True, exist_ok=True)
    ref_code = problem['reference_solution']
    prompt = build_explanation_prompt(problem['prompt'], ref_code)

    t0 = time.monotonic()
    explanation_response = client.generate(teacher, prompt)
    gen_elapsed = time.monotonic() - t0

    explanation_text = explanation_response['message']['content']
    exp_verif = verify_explanation(
        explanation_text,
        ref_code,
        problem.get('reference', {}).get('Expected Time Complexity') if isinstance(problem.get('reference'), dict) else None
    )

    record = {
        'configuration': 'reference_explanation_distillation',
        'teacher': teacher,
        'problem_id': problem['id'],
        'source': problem['source'],
        'difficulty': problem.get('difficulty', 'unknown'),
        'tags': problem.get('tags', []),
        'solution_origin': 'reference_solution',
        'reference_solution': ref_code,
        'explanation_prompt': prompt,
        'explanation_response': explanation_response,
        'explanation': explanation_text,
        'explanation_verification': exp_verif,
        'generation_wall_seconds': gen_elapsed,
        'metadata_hash': digest(metadata),
        'status': 'accepted' if exp_verif['verified'] else 'explanation_rejected',
        'created_at': timestamp()
    }
    atomic_json(record_file, record)
    return record['status']


def solve_config_b(config, client, validator, teacher, problem, metadata,
                   force_retry=False, override_attempts=None, override_budget=None,
                   prompt_options=None):
    """
    Configuration B: Independent Teacher Solution Distillation with exact function signatures,
    feedback-guided retries (up to 3 or 5 attempts), and fail-fast Docker validation.
    """
    record_dir = Path(config['data_dir']) / 'raw' / slug(teacher) / 'config_b' / problem['id']
    record_file = record_dir / 'record.json'

    # Skip previously evaluated solutions to resume from where it left off
    if record_file.exists():
        existing = read_json(record_file)
        if existing.get('accepted') and existing.get('status') == 'accepted':
            return 'already_accepted'
        if not force_retry and not config.get('repair', {}).get('retry_failed_checkpoints', False):
            if existing.get('status') and not existing.get('transient', False):
                return existing.get('status')

    record_dir.mkdir(parents=True, exist_ok=True)

    repair_cfg = config.get('repair', {})
    max_attempts = override_attempts or repair_cfg.get('max_attempts', 3)
    total_budget = override_budget or repair_cfg.get('problem_budget_seconds', 150)
    problem_start = time.monotonic()

    opts = prompt_options or {}
    use_example = bool(opts.get('include_example', False))
    use_cot = bool(opts.get('chain_of_thought', False))
    use_history = bool(opts.get('include_failure_history', False))

    initial_prompt, stub_code = build_initial_prompt(
        problem, include_example=use_example, chain_of_thought=use_cot
    )
    current_prompt = initial_prompt

    attempts_history = []
    prior_failure_summaries = []
    verified_code = None
    last_val_res = None
    terminal_status = 'rejected'

    for attempt in range(1, max_attempts + 1):
        elapsed_so_far = time.monotonic() - problem_start
        if elapsed_so_far >= total_budget:
            terminal_status = 'resource_limited'
            break

        t0 = time.monotonic()
        try:
            gen_response = client.generate(teacher, current_prompt)
        except Exception as e:
            attempts_history.append({
                'attempt': attempt,
                'prompt': current_prompt,
                'error': str(e),
                'wall_seconds': time.monotonic() - t0
            })
            terminal_status = 'generation_error'
            break

        gen_elapsed = time.monotonic() - t0
        raw_content = gen_response['message']['content']

        extracted_code = None
        extraction_error = None
        try:
            extracted_code = extract(raw_content)
        except Exception as e:
            extraction_error = str(e)

        attempt_record = {
            'attempt': attempt,
            'prompt': current_prompt,
            'generation_response': gen_response,
            'extracted_code': extracted_code,
            'extraction_error': extraction_error,
            'generation_wall_seconds': gen_elapsed
        }

        if extraction_error:
            attempt_record['status'] = 'extraction_error'
            attempts_history.append(attempt_record)
            if attempt < max_attempts and (time.monotonic() - problem_start) < total_budget:
                feedback = f"Code extraction failed: {extraction_error}\nPlease return only complete Python code within a single ```python block."
                current_prompt = build_repair_prompt(
                    problem, raw_content, feedback, stub_code, attempt + 1,
                    chain_of_thought=use_cot,
                    prior_failures=prior_failure_summaries if use_history else None
                )
                prior_failure_summaries.append(f"Attempt {attempt}: extraction_error ({extraction_error})")
                continue
            else:
                terminal_status = 'extraction_error'
                break

        # Validate in Docker sandbox with fail-fast
        val_res = validator.validate(extracted_code, problem)
        attempt_record['validation'] = val_res
        attempt_record['status'] = val_res['status']
        attempts_history.append(attempt_record)
        last_val_res = val_res

        if val_res['accepted']:
            verified_code = extracted_code
            terminal_status = 'accepted'
            break
        else:
            terminal_status = val_res['status']
            if val_res['status'] == 'docker_infrastructure_failure':
                # Infrastructure failure: stop retrying as a model failure
                break
            if attempt < max_attempts and (time.monotonic() - problem_start) < total_budget:
                feedback = build_failure_feedback(val_res, problem, stub_code)
                current_prompt = build_repair_prompt(
                    problem, extracted_code, feedback, stub_code, attempt + 1,
                    chain_of_thought=use_cot,
                    prior_failures=prior_failure_summaries if use_history else None
                )
                first_line = feedback.splitlines()[0] if feedback else val_res['status']
                prior_failure_summaries.append(f"Attempt {attempt}: {first_line}")
            else:
                break

    # If code is verified, generate and verify explanation
    exp_prompt = None
    exp_response = None
    exp_text = None
    exp_verif = None
    exp_elapsed = 0.0

    if verified_code:
        exp_prompt = build_explanation_prompt(problem['prompt'], verified_code)
        # Requirement 5: Retry only explanation if explanation fails
        for exp_att in range(1, 3):
            if (time.monotonic() - problem_start) >= total_budget:
                break
            t_exp = time.monotonic()
            try:
                exp_response = client.generate(teacher, exp_prompt)
            except Exception:
                continue
            exp_elapsed = time.monotonic() - t_exp
            exp_text = exp_response['message']['content']
            exp_verif = verify_explanation(exp_text, verified_code)
            if exp_verif['verified']:
                break

    is_accepted = (verified_code is not None) and (exp_verif is not None and exp_verif['verified'])
    final_status = 'accepted' if is_accepted else ('explanation_rejected' if verified_code else terminal_status)

    record = {
        'configuration': 'independent_solution_distillation',
        'teacher': teacher,
        'problem_id': problem['id'],
        'source': problem['source'],
        'difficulty': problem.get('difficulty', 'unknown'),
        'tags': problem.get('tags', []),
        'solution_origin': 'independent_teacher_generation',
        'signature_stub': stub_code,
        'attempts_count': len(attempts_history),
        'attempts': attempts_history,
        'extracted_code': verified_code if verified_code else (attempts_history[-1].get('extracted_code') if attempts_history else None),
        'validation': last_val_res,
        'accepted': is_accepted,
        'status': final_status,
        'total_wall_seconds': time.monotonic() - problem_start,
        'metadata_hash': digest(metadata),
        'created_at': timestamp()
    }
    if force_retry:
        record['consensus_retry'] = True
    if any(opts.values()):
        record['prompt_options'] = {k: bool(v) for k, v in opts.items() if v}
    if is_accepted:
        record['explanation_prompt'] = exp_prompt
        record['explanation_response'] = exp_response
        record['explanation'] = exp_text
        record['explanation_verification'] = exp_verif
        record['explanation_wall_seconds'] = exp_elapsed

    atomic_json(record_file, record)
    return final_status


def export_datasets(config, problems, teachers):
    """Export student-agnostic JSONL datasets for Configuration A and Configuration B."""
    out_dir = Path(config['data_dir']) / 'verified'
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_lookup = {p['id']: p for p in problems}

    summaries = {}
    for teacher in teachers:
        t_slug = slug(teacher)
        summaries[teacher] = {'config_a': {'accepted': 0, 'total': len(problems)},
                              'config_b': {'accepted': 0, 'total': len(problems)}}

        # Export Configuration A
        records_a = []
        provenance_a = []
        for p in problems:
            rf = Path(config['data_dir']) / 'raw' / t_slug / 'config_a' / p['id'] / 'record.json'
            if rf.exists():
                rec = read_json(rf)
                if rec.get('status') == 'accepted':
                    line_no = len(records_a) + 1
                    ref_code = rec['reference_solution']
                    explanation = rec['explanation'].strip()
                    target_completion = f"{explanation}\n\n```python\n{ref_code}```\n"

                    records_a.append({
                        'prompt': p['prompt'],
                        'completion': target_completion,
                        'explanation': explanation,
                        'code': ref_code,
                        'configuration': 'reference_explanation_distillation',
                        'teacher': teacher,
                        'problem_id': p['id'],
                        'source': p['source'],
                        'difficulty': p['difficulty'],
                        'tags': p['tags'],
                        'solution_origin': 'reference_solution'
                    })
                    provenance_a.append({
                        'line': line_no,
                        'problem_id': p['id'],
                        'source': p['source'],
                        'url': p.get('provenance', {}).get('url', ''),
                        'record_path': str(rf.relative_to(config['data_dir']))
                    })
        summaries[teacher]['config_a']['accepted'] = len(records_a)
        if records_a or (Path(config['data_dir']) / 'raw' / t_slug / 'config_a').exists():
            write_jsonl(out_dir / f"{t_slug}_config_a.jsonl", records_a)
            write_jsonl(out_dir / f"{t_slug}_config_a.provenance.jsonl", provenance_a)

        # Export Configuration B
        records_b = []
        provenance_b = []
        for p in problems:
            rf = Path(config['data_dir']) / 'raw' / t_slug / 'config_b' / p['id'] / 'record.json'
            if rf.exists():
                rec = read_json(rf)
                if rec.get('status') == 'accepted':
                    line_no = len(records_b) + 1
                    code = rec['extracted_code']
                    explanation = rec['explanation'].strip()
                    target_completion = f"{explanation}\n\n```python\n{code}```\n"

                    records_b.append({
                        'prompt': p['prompt'],
                        'completion': target_completion,
                        'explanation': explanation,
                        'code': code,
                        'configuration': 'independent_solution_distillation',
                        'teacher': teacher,
                        'problem_id': p['id'],
                        'source': p['source'],
                        'difficulty': p['difficulty'],
                        'tags': p['tags'],
                        'solution_origin': 'independent_teacher_generation'
                    })
                    provenance_b.append({
                        'line': line_no,
                        'problem_id': p['id'],
                        'source': p['source'],
                        'url': p.get('provenance', {}).get('url', ''),
                        'record_path': str(rf.relative_to(config['data_dir']))
                    })
        summaries[teacher]['config_b']['accepted'] = len(records_b)
        if records_b or (Path(config['data_dir']) / 'raw' / t_slug / 'config_b').exists():
            write_jsonl(out_dir / f"{t_slug}_config_b.jsonl", records_b)
            write_jsonl(out_dir / f"{t_slug}_config_b.provenance.jsonl", provenance_b)

    return summaries


def run_pipeline(config, configuration_mode, mode='pilot', model=None):
    """
    Run pipeline for Configuration A ('config_a') or Configuration B ('config_b').
    """
    corpus_file = Path(config['data_dir']) / 'problems' / 'verified_corpus.jsonl'
    if not corpus_file.exists():
        logging.info('Verified corpus not found; building corpus first...')
        build_verified_corpus(config, target_count=config['dataset'].get('count', 1000))

    problems = jsonl(corpus_file)
    teachers = [model] if model else config['teachers']
    if mode == 'smoke':
        # Select up to 5 previously failed problems in config_b for quick verification
        failed_problems = []
        for p in problems:
            rf = Path(config['data_dir']) / 'raw' / slug(teachers[0]) / 'config_b' / p['id'] / 'record.json'
            if rf.exists():
                rec = read_json(rf)
                if not rec.get('accepted'):
                    failed_problems.append(p)
                    if len(failed_problems) == 5:
                        break
        active_problems = failed_problems if failed_problems else problems[:5]
    elif mode == 'pilot':
        active_problems = problems[:config['dataset']['pilot_count']]
    else:
        active_problems = problems
    client = Ollama(config['ollama'])
    validator = DockerValidator(config['validation']) if configuration_mode == 'config_b' else None

    logging.info('Starting %s in mode=%s for %d problems with teachers: %s',
                 configuration_mode, mode, len(active_problems), teachers)

    for teacher in teachers:
        logging.info('Preloading and verifying metadata for teacher: %s', teacher)
        try:
            metadata = client.metadata(teacher)
            client.unload_all()
            # Warm up model to keep first inference within budget
            client.api('/api/generate', {'model': teacher, 'keep_alive': config['ollama']['keep_alive'],
                                         'stream': False, 'options': client.options_for(teacher)})
        except Exception:
            logging.exception('Teacher preflight failed: %s', teacher)
            continue

        try:
            for i, p in enumerate(active_problems, 1):
                try:
                    if configuration_mode == 'config_a':
                        status = solve_config_a(config, client, teacher, p, metadata)
                    else:
                        status = solve_config_b(config, client, validator, teacher, p, metadata)
                except Exception as e:
                    logging.exception('Problem %s failed with exception: %s', p['id'], e)
                    status = 'infrastructure_error'
                logging.info('[%s] %s [%d/%d] %s -> %s', configuration_mode, teacher, i, len(active_problems), p['id'][:10], status)
        finally:
            try:
                client.unload(teacher)
            except Exception:
                pass

    summaries = export_datasets(config, problems, teachers)
    logging.info('Export complete: %s', json.dumps(summaries, indent=2))
    return summaries


def run_consensus_retries(config, max_attempts=5, problem_budget=250, model=None,
                          teachers_list=None, prompt_options=None, consensus_levels=None):
    """
    Identifies problems solved by high multi-model consensus (e.g. 3, 4, or 5 models) in Configuration B,
    and gives models that missed them up to max_attempts (default: 5) to recover them.
    """
    from .consensus import analyze_consensus
    corpus_file = Path(config['data_dir']) / 'problems' / 'verified_corpus.jsonl'
    problems = jsonl(corpus_file)
    corpus_lookup = {p['id']: p for p in problems}

    logging.info("Computing cross-model consensus to identify high-consensus problems...")
    summary = analyze_consensus(config, teachers=teachers_list, target_consensus=consensus_levels)
    targeted_map = summary['targeted_retries']

    teachers = [model] if model else [t for t in summary['teachers'] if len(targeted_map.get(t, [])) > 0]
    client = Ollama(config['ollama'])
    validator = DockerValidator(config['validation'])

    logging.info("Starting targeted consensus retries (max_attempts=%d, budget=%ds, prompt_options=%s) for teachers: %s",
                 max_attempts, problem_budget, prompt_options or {}, teachers)

    total_recovered = 0

    for teacher in teachers:
        missed_list = targeted_map.get(teacher, [])
        if not missed_list:
            logging.info("Teacher %s has 0 missed 3-or-4 consensus problems. Skipping.", teacher)
            continue

        logging.info("Preloading teacher %s for %d targeted retries...", teacher, len(missed_list))
        try:
            metadata = client.metadata(teacher)
            client.unload_all()
            client.api('/api/generate', {'model': teacher, 'keep_alive': config['ollama']['keep_alive']})
        except Exception:
            logging.exception("Teacher preflight failed: %s", teacher)
            continue

        recovered_count = 0
        try:
            for i, p_info in enumerate(missed_list, 1):
                pid = p_info['problem_id']
                p = corpus_lookup[pid]
                try:
                    status = solve_config_b(config, client, validator, teacher, p, metadata,
                                            force_retry=True,
                                            override_attempts=max_attempts,
                                            override_budget=problem_budget,
                                            prompt_options=prompt_options)
                except Exception as e:
                    logging.exception("Retry failed for problem %s: %s", pid, e)
                    status = 'infrastructure_error'

                if status == 'accepted':
                    recovered_count += 1
                    total_recovered += 1

                logging.info('[consensus-retry-%d] %s [%d/%d] %s -> %s (consensus=%d)',
                             max_attempts, teacher, i, len(missed_list), pid[:10], status, p_info['consensus_count'])
        finally:
            try:
                client.unload(teacher)
            except Exception:
                pass

        logging.info("Teacher %s recovered %d / %d high-consensus problems!",
                     teacher, recovered_count, len(missed_list))

    # Re-export all verified datasets
    summaries = export_datasets(config, problems, summary['teachers'])
    logging.info("Updated verified exports complete.")

    # Re-generate empirical and consensus reports
    from .report_v2 import generate_report
    generate_report(config)
    updated_summary = analyze_consensus(config, teachers=teachers_list, target_consensus=consensus_levels)
    logging.info("Consensus retries complete. Total recovered: %d.", total_recovered)
    return updated_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config-v3.yaml')
    sub = parser.add_subparsers(dest='command', required=True)

    prep = sub.add_parser('prepare', help='Build verified Python reference corpus')
    prep.add_argument('--target', type=int, default=1000)

    run_a = sub.add_parser('run-config-a', help='Run Configuration A: Reference-Solution Explanation Distillation')
    run_a.add_argument('--mode', choices=['pilot', 'full'], default='pilot')
    run_a.add_argument('--model')

    run_b = sub.add_parser('run-config-b', help='Run Configuration B: Independent Solution Distillation')
    run_b.add_argument('--mode', choices=['pilot', 'full', 'smoke'], default='pilot')
    run_b.add_argument('--model')

    ret = sub.add_parser('run-consensus-retries', help='Re-evaluate high-consensus problems with 5 attempts')
    ret.add_argument('--max-attempts', type=int, default=5)
    ret.add_argument('--budget', type=int, default=250)
    ret.add_argument('--model')
    ret.add_argument('--teachers', nargs='+', help='Override active teacher list for consensus calculation')
    ret.add_argument('--consensus-levels', nargs='+', type=int, help='Consensus counts to target for retries (e.g. 3 4 5)')
    ret.add_argument('--include-example', action='store_true', help='Include 1 concrete input/output example in prompt')
    ret.add_argument('--cot', action='store_true', help='Ask for brief step-by-step reasoning before a single ```python block')
    ret.add_argument('--retry-history', action='store_true', help='Include compact 1-line summaries of earlier failed attempts')

    con = sub.add_parser('consensus', help='Analyze cross-model solve consensus')
    con.add_argument('--teachers', nargs='+', help='Override active teacher list for consensus calculation')
    con.add_argument('--consensus-levels', nargs='+', type=int, help='Consensus counts to target for retries (e.g. 3 4 5)')
    sub.add_parser('doctor', help='Verify environment, Ollama models, and Docker sandbox')
    sub.add_parser('export', help='Export student-agnostic JSONL files')

    rep = sub.add_parser('report', help='Generate summary reports')
    rep.add_argument('--model')

    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config)
    Path(config['data_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)

    with FileLock(str(Path(config['data_dir']) / '.pipeline.lock'), timeout=0):
        if args.command == 'prepare':
            validator = DockerValidator(config['validation'])
            corpus, manifest = build_verified_corpus(config, target_count=args.target, validator=validator)
            print(f"Verified corpus created with {len(corpus)} problems.")
            print(json.dumps(manifest['audit'], indent=2))

        elif args.command == 'run-config-a':
            summaries = run_pipeline(config, 'config_a', mode=args.mode, model=args.model)
            print(json.dumps(summaries, indent=2))

        elif args.command == 'run-config-b':
            summaries = run_pipeline(config, 'config_b', mode=args.mode, model=args.model)
            print(json.dumps(summaries, indent=2))

        elif args.command == 'run-consensus-retries':
            prompt_opts = {
                'include_example': args.include_example,
                'chain_of_thought': args.cot,
                'include_failure_history': args.retry_history
            }
            updated = run_consensus_retries(
                config,
                max_attempts=args.max_attempts,
                problem_budget=args.budget,
                model=args.model,
                teachers_list=args.teachers,
                prompt_options=prompt_opts,
                consensus_levels=args.consensus_levels
            )
            print(json.dumps(updated['consensus_counts'], indent=2))

        elif args.command == 'consensus':
            from .consensus import analyze_consensus
            summary = analyze_consensus(config, teachers=args.teachers, target_consensus=args.consensus_levels)
            print(json.dumps(summary['consensus_counts'], indent=2))

        elif args.command == 'export':
            corpus_file = Path(config['data_dir']) / 'problems' / 'verified_corpus.jsonl'
            problems = jsonl(corpus_file)
            summaries = export_datasets(config, problems, config['teachers'])
            print(json.dumps(summaries, indent=2))

        elif args.command == 'doctor':
            client = Ollama(config['ollama'])
            print('Ollama version:', client.api('/api/version'))
            for t in config['teachers']:
                meta = client.metadata(t)
                print(f"Teacher {t}: digest={meta['tag']['digest'][:16]} quant={meta['tag']['details']['quantization_level']}")
            validator = DockerValidator(config['validation'])
            print('Docker sandbox image:', validator.image)
            print('Fail-fast enabled:', validator.limits.get('fail_fast'))
            print('Problem budget:', validator.limits.get('problem_seconds'), 'seconds')

        elif args.command == 'report':
            from .report_v2 import generate_report
            generate_report(config, model=args.model)


if __name__ == '__main__':
    main()

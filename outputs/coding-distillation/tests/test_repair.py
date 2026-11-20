"""
Automated unit tests for Configuration B feedback-guided repair,
signature extraction, budget enforcement, and checkpoint categories.
"""

import ast
import tempfile
import time
import unittest
from pathlib import Path

from src.repair import (
    extract_function_signature,
    build_initial_prompt,
    build_failure_feedback,
    build_repair_prompt
)
from src.pipeline_v2 import solve_config_a, solve_config_b, export_datasets
from src.common import atomic_json, read_json


class FakeClient:
    """Mock Ollama client for fast deterministic unit tests."""
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.prompts_received = []

    def generate(self, model, prompt):
        self.call_count += 1
        self.prompts_received.append(prompt)
        if self.responses:
            idx = min(self.call_count - 1, len(self.responses) - 1)
            content = self.responses[idx]
        else:
            content = "```python\ndef test_fn(x):\n    return x\n```"
        return {'message': {'content': content}, 'total_duration': 1000}


class FakeValidator:
    """Mock DockerValidator for deterministic unit tests."""
    def __init__(self, outcomes=None):
        # outcomes: list of dicts or booleans for consecutive attempts
        self.outcomes = outcomes or []
        self.call_count = 0

    def validate(self, code, problem):
        self.call_count += 1
        if self.outcomes:
            idx = min(self.call_count - 1, len(self.outcomes) - 1)
            item = self.outcomes[idx]
            if isinstance(item, dict):
                return item
            elif item is True:
                return {'accepted': True, 'status': 'accepted', 'tests_total': 1, 'tests_passed': 1, 'cases': []}
            else:
                return {'accepted': False, 'status': 'wrong_answer', 'tests_total': 1, 'tests_passed': 0,
                        'cases': [{'passed': False, 'status': 'wrong_answer', 'index': 0, 'expected_output': 42, 'stdout': 0}]}
        return {'accepted': True, 'status': 'accepted', 'tests_total': 1, 'tests_passed': 1, 'cases': []}


class RepairTests(unittest.TestCase):

    def test_signature_extraction(self):
        # 1. Standard FunctionDef
        prob1 = {
            'fn_name': 'has_close_elements',
            'reference_solution': 'def has_close_elements(numbers, threshold):\n    return True\n'
        }
        sig, stub = extract_function_signature(prob1)
        self.assertEqual(sig, 'def has_close_elements(numbers, threshold):')
        self.assertIn('def has_close_elements(numbers, threshold):', stub)
        self.assertIn('pass', stub)

        # 2. Lambda Assignment
        prob2 = {
            'fn_name': 'validate_ean',
            'reference_solution': 'validate_ean = lambda code: len(code) == 13\n'
        }
        sig, stub = extract_function_signature(prob2)
        self.assertEqual(sig, 'def validate_ean(code):')

        # 3. Fallback to inputs count
        prob3 = {
            'fn_name': 'mystery_func',
            'reference_solution': '# no function definition',
            'tests': {'inputs': [[1, 2, 3]]}
        }
        sig, stub = extract_function_signature(prob3)
        self.assertEqual(sig, 'def mystery_func(arg1, arg2, arg3):')

        # 4. Prompt embedding
        prob4 = {
            'prompt': 'Write a function to check elements.',
            'fn_name': 'has_close_elements',
            'reference_solution': 'def has_close_elements(numbers, threshold):\n    pass\n'
        }
        prompt, stub = build_initial_prompt(prob4)
        self.assertIn('def has_close_elements(numbers, threshold):', prompt)
        self.assertIn('Write a function to check elements.', prompt)

    def test_retry_prompt_feedback(self):
        problem = {
            'prompt': 'Double the input number.',
            'question': 'Double the input number.',
            'fn_name': 'double_num',
            'tests': {'inputs': [[5]]}
        }
        stub = 'def double_num(x):\n    pass'

        # Wrong answer feedback
        val_wa = {
            'status': 'wrong_answer',
            'cases': [{'passed': False, 'status': 'wrong_answer', 'index': 0, 'expected_output': 10, 'stdout': 7}]
        }
        fb = build_failure_feedback(val_wa, problem, stub)
        self.assertIn('wrong_answer', fb)
        self.assertIn('10', fb)
        self.assertIn('7', fb)

        # CPU limit feedback
        val_cpu = {
            'status': 'cpu_limit',
            'cases': [{'passed': False, 'status': 'cpu_limit', 'index': 0}]
        }
        fb_cpu = build_failure_feedback(val_cpu, problem, stub)
        self.assertIn('cpu_limit', fb_cpu)
        self.assertIn('optimize time complexity', fb_cpu.lower())

        # Runtime error feedback
        val_rt = {
            'status': 'runtime_error',
            'cases': [{'passed': False, 'status': 'runtime_error', 'index': 0, 'stderr': 'Traceback:\nNameError: name foo is not defined'}]
        }
        fb_rt = build_failure_feedback(val_rt, problem, stub)
        self.assertIn('runtime_error', fb_rt)
        self.assertIn('NameError', fb_rt)

        # Build repair prompt
        repair_prompt = build_repair_prompt(problem, 'def double_num(x):\n    return x + 2', fb, stub, attempt_num=2)
        self.assertIn('Attempt 1 failed testing', repair_prompt)
        self.assertIn('return x + 2', repair_prompt)
        self.assertIn('def double_num(x):', repair_prompt)

    def test_max_three_attempts_enforced(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'data_dir': tmpdir,
                'repair': {'max_attempts': 3, 'problem_budget_seconds': 180, 'retry_failed_checkpoints': True}
            }
            # Model returns wrong code 3 times
            client = FakeClient([
                "```python\ndef double_num(x):\n    return x + 1\n```",
                "```python\ndef double_num(x):\n    return x + 2\n```",
                "```python\ndef double_num(x):\n    return x + 3\n```",
                "```python\ndef double_num(x):\n    return x * 2\n```" # 4th should NOT be called
            ])
            validator = FakeValidator([False, False, False, True])
            problem = {
                'id': 'prob_retry_max',
                'source': 'test',
                'prompt': 'Double x',
                'fn_name': 'double_num',
                'tests': {'inputs': [[5]], 'outputs': [10]}
            }

            status = solve_config_b(config, client, validator, 'test_teacher', problem, {})
            self.assertEqual(status, 'wrong_answer')
            self.assertEqual(client.call_count, 3) # Exactly 3 attempts, 4th never called!
            self.assertEqual(validator.call_count, 3)

            rec = read_json(Path(tmpdir) / 'raw' / 'test_teacher' / 'config_b' / 'prob_retry_max' / 'record.json')
            self.assertEqual(rec['attempts_count'], 3)
            self.assertFalse(rec['accepted'])

    def test_max_two_attempts_enforced(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'data_dir': tmpdir,
                'repair': {'max_attempts': 2, 'problem_budget_seconds': 75}
            }
            # Model returns wrong code 3 times
            client = FakeClient([
                "```python\ndef double_num(x):\n    return x + 1\n```",
                "```python\ndef double_num(x):\n    return x + 2\n```",
                "```python\ndef double_num(x):\n    return x * 2\n```" # 3rd should NOT be called
            ])
            validator = FakeValidator([False, False, True])
            problem = {
                'id': 'prob_retry_two_max',
                'source': 'test',
                'prompt': 'Double x',
                'fn_name': 'double_num',
                'tests': {'inputs': [[5]], 'outputs': [10]}
            }

            status = solve_config_b(config, client, validator, 'test_teacher', problem, {})
            self.assertEqual(status, 'wrong_answer')
            self.assertEqual(client.call_count, 2) # Exactly 2 attempts, 3rd never called!
            self.assertEqual(validator.call_count, 2)

            rec = read_json(Path(tmpdir) / 'raw' / 'test_teacher' / 'config_b' / 'prob_retry_two_max' / 'record.json')
            self.assertEqual(rec['attempts_count'], 2)
            self.assertFalse(rec['accepted'])

    def test_second_attempt_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'data_dir': tmpdir,
                'repair': {'max_attempts': 3, 'problem_budget_seconds': 180, 'retry_failed_checkpoints': True}
            }
            # Attempt 1 fails, Attempt 2 succeeds, then explanation succeeds
            exp_text = ("Algorithm selection and overall approach:\nSimple doubling.\n\n"
                        "Important steps and data structures used:\nMultiply by 2.\n\n"
                        "Justification for correctness and boundary conditions:\nHandles all integers.\n\n"
                        "Time complexity: O(1)\nAuxiliary space complexity: O(1)")
            client = FakeClient([
                "```python\ndef double_num(x):\n    return x + 1\n```",
                "```python\ndef double_num(x):\n    return x * 2\n```",
                exp_text
            ])
            validator = FakeValidator([False, True])
            problem = {
                'id': 'prob_retry_success',
                'source': 'test',
                'prompt': 'Double x',
                'fn_name': 'double_num',
                'tests': {'inputs': [[5]], 'outputs': [10]}
            }

            status = solve_config_b(config, client, validator, 'test_teacher', problem, {})
            self.assertEqual(status, 'accepted')
            self.assertEqual(validator.call_count, 2)

            rec = read_json(Path(tmpdir) / 'raw' / 'test_teacher' / 'config_b' / 'prob_retry_success' / 'record.json')
            self.assertTrue(rec['accepted'])
            self.assertEqual(rec['attempts_count'], 2)

    def test_checkpoint_categories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'data_dir': tmpdir,
                'repair': {'max_attempts': 3, 'problem_budget_seconds': 180, 'retry_failed_checkpoints': True}
            }
            client = FakeClient()
            validator = FakeValidator()

            # Category 1: Previously accepted problem
            acc_dir = Path(tmpdir) / 'raw' / 'test_teacher' / 'config_b' / 'prob_acc'
            acc_dir.mkdir(parents=True)
            atomic_json(acc_dir / 'record.json', {'accepted': True, 'status': 'accepted', 'extracted_code': 'def f(): pass'})
            prob_acc = {'id': 'prob_acc', 'source': 'test', 'prompt': 'test'}

            res_acc = solve_config_b(config, client, validator, 'test_teacher', prob_acc, {})
            self.assertEqual(res_acc, 'already_accepted')
            self.assertEqual(client.call_count, 0) # Skipped completely without model calls!

            # Category 2: Previously failed problem
            fail_dir = Path(tmpdir) / 'raw' / 'test_teacher' / 'config_b' / 'prob_fail'
            fail_dir.mkdir(parents=True)
            atomic_json(fail_dir / 'record.json', {'accepted': False, 'status': 'wrong_answer'})
            prob_fail = {
                'id': 'prob_fail',
                'source': 'test',
                'prompt': 'test',
                'fn_name': 'f',
                'tests': {'inputs': [[1]], 'outputs': [1]}
            }

            exp_text = ("Algorithm selection and overall approach:\nReturn x.\n\n"
                        "Important steps and data structures used:\nReturn statement.\n\n"
                        "Justification for correctness and boundary conditions:\nExact match.\n\n"
                        "Time complexity: O(1)\nAuxiliary space complexity: O(1)")
            client.responses = ["```python\ndef f(x):\n    return x\n```", exp_text]
            res_fail = solve_config_b(config, client, validator, 'test_teacher', prob_fail, {})
            self.assertEqual(res_fail, 'accepted')
            self.assertGreater(client.call_count, 0) # Reprocessed!

            # Category 3: Default behavior without retry_failed_checkpoints skips failed problems
            default_config = {'data_dir': tmpdir, 'repair': {}}
            skip_dir = Path(tmpdir) / 'raw' / 'test_teacher' / 'config_b' / 'prob_skip'
            skip_dir.mkdir(parents=True)
            atomic_json(skip_dir / 'record.json', {'accepted': False, 'status': 'wrong_answer'})
            client.call_count = 0
            res_skip = solve_config_b(default_config, client, validator, 'test_teacher', {'id': 'prob_skip', 'source': 'test', 'prompt': 'test'}, {})
            self.assertEqual(res_skip, 'wrong_answer')
            self.assertEqual(client.call_count, 0) # Skipped without calling model!

            # Category 4: Force retry with 5 attempts on targeted high-consensus problem
            client.call_count = 0
            client.responses = [
                "```python\ndef f(x):\n    return x + 1\n```",
                "```python\ndef f(x):\n    return x + 2\n```",
                "```python\ndef f(x):\n    return x + 3\n```",
                "```python\ndef f(x):\n    return x + 4\n```",
                "```python\ndef f(x):\n    return x\n```", # 5th attempt succeeds!
                exp_text
            ]
            validator = FakeValidator([False, False, False, False, True])
            prob_retry5 = {
                'id': 'prob_skip',
                'source': 'test',
                'prompt': 'test',
                'fn_name': 'f',
                'tests': {'inputs': [[1]], 'outputs': [1]}
            }
            res_retry5 = solve_config_b(default_config, client, validator, 'test_teacher', prob_retry5, {},
                                        force_retry=True, override_attempts=5, override_budget=250)
            self.assertEqual(res_retry5, 'accepted')
            self.assertEqual(validator.call_count, 5) # All 5 attempts tested!
            rec5 = read_json(skip_dir / 'record.json')
            self.assertTrue(rec5['accepted'])
            self.assertTrue(rec5['consensus_retry'])
            self.assertEqual(rec5['attempts_count'], 5)

    def test_export_no_duplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {'data_dir': tmpdir}
            t_slug = 'test_teacher'

            # Save two distinct problems
            for pid in ['prob1', 'prob2']:
                p_dir = Path(tmpdir) / 'raw' / t_slug / 'config_b' / pid
                p_dir.mkdir(parents=True)
                atomic_json(p_dir / 'record.json', {
                    'accepted': True,
                    'status': 'accepted',
                    'extracted_code': f'def f_{pid}(): pass',
                    'explanation': 'Valid explanation'
                })

            problems = [
                {'id': 'prob1', 'prompt': 'p1', 'source': 's', 'difficulty': 'easy', 'tags': []},
                {'id': 'prob2', 'prompt': 'p2', 'source': 's', 'difficulty': 'easy', 'tags': []}
            ]

            summaries = export_datasets(config, problems, ['test_teacher'])
            self.assertEqual(summaries['test_teacher']['config_b']['accepted'], 2)

            # Read exported JSONL
            out_file = Path(tmpdir) / 'verified' / f'{t_slug}_config_b.jsonl'
            with open(out_file, encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            self.assertEqual(len(lines), 2)

    def test_optional_prompt_engineering_flags(self):
        problem = {
            'id': 'prob_pe',
            'source': 'test',
            'prompt': 'Return double of x.',
            'fn_name': 'double_x',
            'reference_solution': 'def double_x(x):\n    return x * 2\n',
            'tests': {'inputs': [[5]], 'outputs': [10]}
        }

        # Default: neither example nor CoT present
        default_p, stub = build_initial_prompt(problem)
        self.assertNotIn('Example input/output:', default_p)
        self.assertNotIn('think step-by-step', default_p)

        # Enabled: both example I/O and CoT present
        enhanced_p, _ = build_initial_prompt(problem, include_example=True, chain_of_thought=True)
        self.assertIn('Example input/output:', enhanced_p)
        self.assertIn('[5]', enhanced_p)
        self.assertIn('10', enhanced_p)
        self.assertIn('think step-by-step', enhanced_p)

        # Repair prompt with history + CoT
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {'data_dir': tmpdir, 'repair': {'max_attempts': 3, 'problem_budget_seconds': 150}}
            exp_text = ("Algorithm selection and overall approach:\nMultiply by 2.\n\n"
                        "Important steps and data structures used:\nArithmetic.\n\n"
                        "Justification for correctness and boundary conditions:\nHandles ints.\n\n"
                        "Time complexity: O(1)\nAuxiliary space complexity: O(1)")
            client = FakeClient([
                "Think: add 1.\n```python\ndef double_x(x):\n    return x + 1\n```",
                "Think: add 2.\n```python\ndef double_x(x):\n    return x + 2\n```",
                "Think: multiply by 2.\n```python\ndef double_x(x):\n    return x * 2\n```",
                exp_text
            ])
            validator = FakeValidator([False, False, True])
            opts = {'include_example': True, 'chain_of_thought': True, 'include_failure_history': True}
            status = solve_config_b(config, client, validator, 'test_teacher', problem, {},
                                    force_retry=True, override_attempts=3, prompt_options=opts)
            self.assertEqual(status, 'accepted')
            # Attempt 1 prompt had example + CoT
            self.assertIn('Example input/output:', client.prompts_received[0])
            self.assertIn('think step-by-step', client.prompts_received[0])
            # Attempt 3 prompt had prior failure history from Attempt 1
            self.assertIn('Summary of earlier failed attempts', client.prompts_received[2])
            self.assertIn('Attempt 1:', client.prompts_received[2])


if __name__ == '__main__':
    unittest.main()

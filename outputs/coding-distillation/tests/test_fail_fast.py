import time
import unittest
from pathlib import Path

from src.common import jsonl, read_json
from src.pipeline import load_config
from src.validate import DockerValidator

ROOT = Path(__file__).resolve().parents[1]


class FailFastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / 'config-v2.yaml')
        cls.validator = DockerValidator(cls.config['validation'])
        cls.problems = {p['id']: p for p in jsonl(ROOT / 'data-v2/problems/selected.jsonl')}

    def test_problem_8_fail_fast_speedup(self):
        pid = 'c08f71200ec6039c15af56632348719cab48d5a3189f53b02b61b3f7451a86b4'
        raw = read_json(ROOT / f'data-v2/raw/qwen2.5_7b-instruct-fp16/{pid}/attempt-01.json')
        p = self.problems[pid]
        start = time.monotonic()
        res = self.validator.validate(raw['code'], p)
        elapsed = time.monotonic() - start
        self.assertFalse(res['accepted'])
        self.assertIn(res['status'], ('cpu_limit', 'wrong_answer'))
        self.assertLess(elapsed, 15.0)  # was 705 seconds!
        self.assertLess(res['tests_run'], res['tests_total'])
        print(f"\nProblem 8 fail-fast: status={res['status']}, tests_run={res['tests_run']}/{res['tests_total']} in {elapsed:.2f}s (was 705s)")

    def test_problem_15_fail_fast_speedup(self):
        pid = '5a665416ebf406280246f8b04540bdeacb152de06f1182c9c2552a58e2e36c3e'
        raw = read_json(ROOT / f'data-v2/raw/qwen2.5_7b-instruct-fp16/{pid}/attempt-01.json')
        p = self.problems[pid]
        start = time.monotonic()
        res = self.validator.validate(raw['code'], p)
        elapsed = time.monotonic() - start
        self.assertFalse(res['accepted'])
        self.assertIn(res['status'], ('cpu_limit', 'wrong_answer'))
        self.assertLess(elapsed, 15.0)  # was 159 seconds!
        self.assertLess(res['tests_run'], res['tests_total'])
        print(f"\nProblem 15 fail-fast: status={res['status']}, tests_run={res['tests_run']}/{res['tests_total']} in {elapsed:.2f}s (was 159s)")

    def test_infinite_loop_rejected_promptly(self):
        p = next(iter(self.problems.values()))
        start = time.monotonic()
        res = self.validator.validate('while True: pass', p)
        elapsed = time.monotonic() - start
        self.assertFalse(res['accepted'])
        self.assertEqual(res['status'], 'cpu_limit')
        self.assertLess(elapsed, 10.0)
        self.assertEqual(res['tests_run'], 1)

    def test_known_correct_solution_passes_all(self):
        # Problem 4 (sum_it_up) was accepted in the pilot
        pid = '6c17a98aa49e6e67381621a73d5d49d4e605a02c3a3b28d503dec844b2aed125'
        raw = read_json(ROOT / f'data-v2/raw/qwen2.5_7b-instruct-fp16/{pid}/attempt-01.json')
        p = self.problems[pid]
        res = self.validator.validate(raw['code'], p)
        self.assertTrue(res['accepted'])
        self.assertEqual(res['status'], 'accepted')
        self.assertEqual(res['tests_passed'], res['tests_total'])
        self.assertEqual(res['tests_run'], res['tests_total'])


if __name__ == '__main__':
    unittest.main()

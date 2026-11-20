"""Opt in: $env:DISTILL_DOCKER_TESTS='1'; python -m unittest discover -s tests -v"""
import os
import unittest
from pathlib import Path

from src.common import jsonl
from src.dataset import problem
from src.pipeline import load_config
from src.validate import DockerValidator

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('DISTILL_DOCKER_TESTS') == '1', 'Docker opt-in not set')
class DockerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(ROOT / 'config-v2.yaml')['validation']
        cls.cfg.update(wall_seconds=2, cpu_seconds=1)
        cls.v = DockerValidator(cls.cfg)

    def test_stdio(self):
        p = problem(jsonl(ROOT / 'tests/fixtures/problems.jsonl')[0], 0)
        self.assertTrue(self.v.validate('print(sum(map(int,input().split())))', p)['accepted'])

    def test_callable(self):
        p = problem(jsonl(ROOT / 'tests/fixtures/problems.jsonl')[1], 1)
        self.assertTrue(self.v.validate('class Solution:\n def reverseList(self, values):\n  return values[::-1]', p)['accepted'])

    def test_wrong_answer(self):
        p = problem(jsonl(ROOT / 'tests/fixtures/problems.jsonl')[0], 0)
        self.assertFalse(self.v.validate('print(0)', p)['accepted'])

    def test_timeout(self):
        r = self.v.case('import time\ntime.sleep(20)', '', None)
        self.assertEqual(r['status'], 'execution_timeout')

    def test_cpu(self):
        self.assertNotEqual(self.v.case('while True: pass', '', None)['status'], 'ok')

    def test_output_bomb(self):
        self.assertNotEqual(self.v.case('while True: print("x"*100000)', '', None)['status'], 'ok')

    def test_memory_bomb(self):
        self.assertNotEqual(self.v.case('x = bytearray(1024**3)', '', None)['status'], 'ok')

    def test_privilege_and_filesystem(self):
        code = 'import os\nprint(os.getuid())\nopen("/sandbox/runner.py", "w").write("bad")'
        result = self.v.case(code, '', None)
        self.assertNotEqual(result['status'], 'ok')
        self.assertIn('65534', result['stdout'])

    def test_network(self):
        result = self.v.case('import socket\nsocket.create_connection(("1.1.1.1",80),timeout=1)', '', None)
        self.assertNotEqual(result['status'], 'ok')

    def test_signature(self):
        self.assertEqual(self.v.case('def f(a): return a', [1, 2], 'f')['status'], 'incompatible_function_signature')

    def test_fork_denied(self):
        self.assertEqual(self.v.case('import os\nos.fork()', '', None)['status'], 'runtime_error')

    def test_scratch_reset_and_per_case_rss(self):
        from src.sandbox_client import Session
        session = Session(self.v.image, self.cfg)
        try:
            session.initialize('import os\nprint(os.listdir("."))\nopen("x", "w").write("x")', None)
            first = session.case('', 0)
            second = session.case('', 1)
            self.assertEqual(first['stdout'], '[]\n')
            self.assertEqual(second['stdout'], '[]\n')
            self.assertEqual(second['status'], 'ok')
        finally:
            session.close()


if __name__ == '__main__':
    unittest.main()

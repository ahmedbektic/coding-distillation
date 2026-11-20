import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common import atomic_json, jsonl, read_json, slug
from src.dataset import labels, problem, select, prepare
from src.export import export
from src.pipeline import load_config, solve
from src.validate import compare, docker_command, equivalent, extract

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/problems.jsonl'


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = load_config(ROOT / 'config.yaml')
        self.config['data_dir'] = str(Path(self.temp.name) / 'data')
        self.config['results_dir'] = str(Path(self.temp.name) / 'results')
        self.config['dataset']['count'] = 2
        self.rows = jsonl(FIXTURE)
        self.p = problem(self.rows[0], 0)

    def test_extract_keeps_entrypoint(self):
        code = 'import sys\nif __name__ == "__main__":\n    print(sum(map(int, sys.stdin.read().split())))\n'
        self.assertEqual(extract('Here is code:\n```python\n' + code + '```\nExplanation.'), code)
        self.assertEqual(extract(code), code)

    def test_extract_rejects_ambiguous_or_prose(self):
        for value in ('```py\nprint(1)\n```\n```py\nprint(2)\n```', 'Here is the answer:\nprint(1)', '', '<s>print(1)</s>'):
            with self.assertRaises((ValueError, SyntaxError)):
                extract(value)

    def test_extract_does_not_execute(self):
        marker = Path(self.temp.name) / 'executed'
        extract(f'open({str(marker)!r}, "w").write("bad")')
        self.assertFalse(marker.exists())

    def test_no_tests_rejected(self):
        row = dict(self.rows[0], input_output={'inputs': [], 'outputs': []})
        with self.assertRaises(ValueError):
            problem(row, 0)

    def test_unsupported_stdio_rejected(self):
        row = dict(self.rows[0], input_output={'inputs': [[], []], 'outputs': [[], []]})
        with self.assertRaises(ValueError):
            problem(row, 0)

    def test_train_dedup_and_eval_exclusion(self):
        duplicate = dict(self.rows[0], question=self.rows[0]['question'].upper())
        selected, audit = select(self.rows + [duplicate], [self.rows[1]], 1, 1729)
        self.assertEqual(len(selected), 1)
        self.assertEqual(audit['duplicate'], 1)
        self.assertEqual(audit['evaluation_overlap'], 1)

    def test_selection_reproducible(self):
        self.assertEqual(select(self.rows, [], 2, 1), select(self.rows, [], 2, 1))

    def test_real_taco_tags_and_output_envelope(self):
        self.assertEqual(labels("['Sorting', 'Greedy algorithms']"), ['Greedy algorithms', 'Sorting'])
        row = dict(self.rows[1], source='codewars', input_output={
            'fn_name': 'f', 'inputs': [[1], [2]], 'outputs': [[[1]], [[2, 3]]]})
        self.assertEqual(problem(row, 0)['tests']['outputs'], [[1], [2, 3]])

    def test_unsupported_problem_types(self):
        for row in (dict(self.rows[0], question='This is an interactive problem.'),
                    dict(self.rows[1], source='geeksforgeeks')):
            with self.assertRaises(ValueError):
                problem(row, 0)

    def test_prepare_manifest_and_drift(self):
        first = prepare(self.config, FIXTURE)
        self.assertEqual(first, prepare(self.config, FIXTURE))
        self.config['seed'] += 1
        with self.assertRaises(ValueError):
            prepare(self.config, FIXTURE)

    def test_compare_is_not_permissive(self):
        cfg = self.config['validation']
        self.assertTrue(compare(' 1\n2\n', '1 2', None, cfg))
        self.assertFalse(compare('2 1', '1 2', None, cfg))
        self.assertFalse(compare('True', 'true', None, cfg))
        self.assertFalse(equivalent(True, 1))
        self.assertFalse(equivalent(1000000001, 1000000000))
        self.assertFalse(compare('[1]', 1, 'f', cfg))
        self.assertTrue(compare('1.00000001', 1.0, 'f', cfg))
        self.assertFalse(compare('NaN', 1.0, 'f', cfg))

    def test_sandbox_flags(self):
        cmd = docker_command('test', 'sha256:123', self.config['validation'])
        for value in ('--read-only', '--network', 'none', '--cap-drop', 'ALL', '--memory-swap', '--pids-limit', '--cpus', '--tmpfs'):
            self.assertIn(value, cmd)
        self.assertNotIn('--privileged', cmd)
        self.assertNotIn('--volume', cmd)

    def test_checkpoint_resume_and_export(self):
        class Client:
            calls = 0
            def generate(self, *args):
                self.calls += 1
                return {'done': True, 'done_reason': 'stop', 'prompt_eval_count': 100,
                        'eval_count': 10, 'message': {'content': 'print(sum(map(int,input().split())))'}}
        class Validator:
            calls = 0
            def validate(self, *args):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError('simulated crash after raw save')
                return {'accepted': True}
        c, v = Client(), Validator()
        teacher = self.config['teachers'][0]
        with self.assertRaises(RuntimeError):
            solve(self.config, c, v, teacher, self.p, {})
        self.assertEqual(solve(self.config, c, v, teacher, self.p, {}), 'accepted')
        self.assertEqual(solve(self.config, c, v, teacher, self.p, {}), 'accepted')
        self.assertEqual(c.calls, 1)
        report = export(self.config, [self.p, problem(self.rows[1], 1)], [teacher])
        self.assertEqual(report[teacher]['solved'], 1)
        self.assertEqual(report[teacher]['common_problems'], 2)
        output = jsonl(Path(self.config['data_dir']) / 'verified' / (slug(teacher)+'.jsonl'))
        self.assertEqual(set(output[0]), {'prompt', 'completion'})

    def test_bounded_retries_preserve_all_attempts(self):
        class Client:
            calls = 0
            def generate(self, *args):
                self.calls += 1
                raise TimeoutError('offline')
        c = Client()
        with patch('src.pipeline.time.sleep'):
            self.assertEqual(solve(self.config, c, None, 'teacher', self.p, {}), 'generation_failed')
            self.assertEqual(solve(self.config, c, None, 'teacher', self.p, {}), 'generation_failed')
        self.assertEqual(c.calls, 3)
        folder = Path(self.config['data_dir']) / 'raw/teacher' / self.p['id']
        self.assertEqual(len(list(folder.glob('attempt-*.json'))), 3)

    def test_truncation_not_exported(self):
        class Client:
            def generate(self, *args):
                return {'done': True, 'done_reason': 'length', 'message': {'content': 'print(1)'}}
        self.assertEqual(solve(self.config, Client(), None, 't', self.p, {}), 'extraction_failed')


if __name__ == '__main__':
    unittest.main()

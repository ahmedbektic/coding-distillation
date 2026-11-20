import unittest
from src.checkers import judge_special, support_issues, invalid_test_indices


class CheckerTests(unittest.TestCase):
    def test_whiteboard_accepts_alternatives_rejects_invalid(self):
        self.assertTrue(judge_special('cf1430c_operations', '2\n4 2\n3 3\n1 3\n', '1\n4\n'))
        self.assertTrue(judge_special('cf1430c_operations', '2\n4 3\n4 2\n3 1\n', '1\n4\n'))
        self.assertFalse(judge_special('cf1430c_operations', '2\n1 2\n1 3\n2 3\n', '1\n4\n'))
        self.assertFalse(judge_special('cf1430c_operations', '2\n4 4\n4 2\n3 1\n', '1\n4\n'))

    def test_nonpalindrome(self):
        self.assertTrue(judge_special('cf1093b_nonpalindrome', 'baa\n-1', '2\naab\naaa'))
        self.assertFalse(judge_special('cf1093b_nonpalindrome', 'aba\n-1', '2\naab\naaa'))
        self.assertFalse(judge_special('cf1093b_nonpalindrome', 'abc\n-1', '2\naab\naaa'))

    def test_partition(self):
        self.assertTrue(judge_special('cf469d_partition', 'YES 0 0 1 1', '4 5 9\n2 3 4 5'))
        self.assertFalse(judge_special('cf469d_partition', 'YES 0 1 1 1', '4 5 9\n2 3 4 5'))
        self.assertFalse(judge_special('cf469d_partition', 'NO', '4 5 9\n2 3 4 5', 'YES 0 0 1 1'))

    def test_swap_legality_and_final_state(self):
        self.assertTrue(judge_special('cf1148c_swaps', '2 1 3 2 4', '4\n3 4 1 2'))
        self.assertFalse(judge_special('cf1148c_swaps', '1 1 2', '4\n2 1 3 4'))
        self.assertFalse(judge_special('cf1148c_swaps', '0', '4\n3 4 1 2'))

    def test_missing_gfg_driver(self):
        p = {'source': 'geeksforgeeks', 'starter_code': 'def solve(x): pass',
             'tests': {'inputs': ['x'], 'outputs': ['y'], 'fn_name': None}}
        self.assertIn('missing_callable_driver', support_issues(p))

    def test_out_of_spec_is_not_silently_dropped(self):
        p = {'url': 'https://atcoder.jp/contests/abc163/tasks/abc163_d',
             'tests': {'inputs': ['3 2', '10931 60408'], 'outputs': ['10', '0']}}
        self.assertEqual(invalid_test_indices(p), [1])
        self.assertEqual(len(p['tests']['inputs']), 2)

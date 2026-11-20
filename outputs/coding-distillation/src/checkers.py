"""Trusted host-side output judges. Candidate code is never loaded here."""
import collections
import re


def checker_name(problem):
    url = problem.get('url') or ''
    if url.rstrip('/').endswith('/1430/C'):
        return 'cf1430c_operations'
    if url.rstrip('/').endswith('/1093/B'):
        return 'cf1093b_nonpalindrome'
    if url.rstrip('/').endswith('/469/D'):
        return 'cf469d_partition'
    if url.rstrip('/').endswith('/1148/C'):
        return 'cf1148c_swaps'
    return 'provided_output'


def judge_special(name, stdout, input_value, expected=None):
    try:
        if name == 'cf1430c_operations':
            values = list(map(int, input_value.split()))
            tokens = iter(map(int, stdout.split()))
            if values[0] != len(values)-1:
                return False
            for n in values[1:]:
                # For n >= 2 the minimum attainable result is 2.
                if next(tokens) != 2:
                    return False
                counts = collections.Counter(range(1, n+1))
                for _ in range(n-1):
                    a, b = next(tokens), next(tokens)
                    if counts[a] <= 0:
                        return False
                    counts[a] -= 1
                    if counts[b] <= 0:
                        return False
                    counts[b] -= 1
                    counts[(a+b+1)//2] += 1
                if counts[2] != 1 or sum(counts.values()) != 1:
                    return False
            return next(tokens, None) is None
        if name == 'cf1093b_nonpalindrome':
            inputs = input_value.split()
            outputs = stdout.split()
            if int(inputs[0]) != len(outputs) or len(inputs)-1 != len(outputs):
                return False
            return all((out == '-1' and len(set(word)) == 1) or
                       (out != '-1' and out != out[::-1] and collections.Counter(out) == collections.Counter(word))
                       for word, out in zip(inputs[1:], outputs))
        if name == 'cf469d_partition':
            values = list(map(int, input_value.split()))
            n, a, b = values[:3]
            numbers = values[3:]
            output = stdout.split()
            if output == ['NO']:
                # Infeasibility is certified by the dataset oracle, not a guessed search.
                return expected is not None and expected.split() == ['NO']
            if len(output) != n+1 or output[0] != 'YES' or len(numbers) != n:
                return False
            groups = list(map(int, output[1:]))
            if any(g not in (0, 1) for g in groups):
                return False
            assigned = dict(zip(numbers, groups))
            return all(assigned.get((a if g == 0 else b)-x) == g for x, g in assigned.items())
        if name == 'cf1148c_swaps':
            values = list(map(int, input_value.split()))
            n, permutation = values[0], values[1:]
            output = list(map(int, stdout.split()))
            m = output[0]
            if len(permutation) != n or not 0 <= m <= 5*n or len(output) != 1+2*m:
                return False
            for k in range(m):
                a, b = output[1+2*k:3+2*k]
                if not (1 <= a <= n and 1 <= b <= n and 2*abs(a-b) >= n):
                    return False
                permutation[a-1], permutation[b-1] = permutation[b-1], permutation[a-1]
            return permutation == list(range(1, n+1))
    except (ValueError, StopIteration, IndexError):
        return False
    raise ValueError('Unknown checker')


def support_issues(problem):
    tests = problem.get('tests') or {}
    ins, outs, fn = tests.get('inputs'), tests.get('outputs'), tests.get('fn_name')
    issues = []
    if not isinstance(ins, list) or not isinstance(outs, list) or not ins or len(ins) != len(outs):
        return ['unsupported_test_format']
    if fn:
        if not isinstance(fn, str) or not fn.isidentifier() or not all(isinstance(v, list) for v in ins):
            issues.append('unsupported_test_format')
    elif not all(isinstance(v, str) for v in ins + outs):
        issues.append('unsupported_test_format')
    q = problem.get('question', '')
    # Explicit manual review: RGAME asks for one integer sum, despite its constructive tag.
    reviewed_unique = (problem.get('url') or '').rstrip('/') == 'https://www.codechef.com/problems/RGAME'
    if not reviewed_unique and checker_name(problem) == 'provided_output' and (
        'Constructive algorithms' in problem.get('tags', []) or
        re.search(r'(?:print|output|return)\s+(?:any|an arbitrary)|(?:multiple|several)\s+(?:valid |possible |correct )?(?:answers|solutions)|any\s+(?:of them|valid solution)', q, re.I)
    ):
        issues.append('custom_checker_review_required')
    if re.search(r'\b(?:TreeNode|ListNode)\b', problem.get('starter_code', '')):
        issues.append('platform_type_adapter_required')
    if not fn and re.search(r"[\w-]+\.(?:in|out)\b", q):
        issues.append('file_io_review_required')
    if not fn and problem.get('source') == 'geeksforgeeks' and re.search(r'\bdef\s+\w+', problem.get('starter_code', '')):
        issues.append('missing_callable_driver')
    return issues


def invalid_test_indices(problem):
    """Explicit constraint checks for reviewed problems, not a generic constraint parser."""
    url = (problem.get('url') or '').rstrip('/')
    if url.endswith('/1148/C'):
        invalid = []
        for i, value in enumerate(problem['tests']['inputs']):
            try:
                values = list(map(int, value.split()))
                n, permutation = values[0], values[1:]
                if not (2 <= n <= 300000 and n % 2 == 0 and len(permutation) == n and
                        sorted(permutation) == list(range(1, n+1))):
                    invalid.append(i)
            except (ValueError, IndexError, AttributeError):
                invalid.append(i)
        return invalid
    if url.endswith('/abc163_d'):
        invalid = []
        for i, value in enumerate(problem['tests']['inputs']):
            try:
                n, k = map(int, value.split())
                if not (1 <= n <= 200000 and 1 <= k <= n+1):
                    invalid.append(i)
            except (ValueError, AttributeError):
                invalid.append(i)
        return invalid
    return []

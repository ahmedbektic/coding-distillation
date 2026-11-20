import ast
import json
import math
import re
import subprocess
import time
import uuid


def extract(raw):
    """Remove one Markdown wrapper only. Never repair, concatenate, or rewrite code."""
    fences = re.findall(r'```([^\n]*)\n(.*?)```', raw, re.S)
    if fences:
        candidates = [body for language, body in fences if language.strip().lower() in ('', 'python', 'python3', 'py')]
        if len(candidates) != 1 or len(fences) != 1:
            raise ValueError('ambiguous code blocks')
        code = candidates[0]
    else:
        code = raw
    code = code.replace('\r\n', '\n').strip('\r\n') + '\n'
    if not code.strip():
        raise ValueError('empty code')
    if re.search(r'<\|[^\n]*?\|>|\[/?INST\]|<</?SYS>>|</?s>|</?think>', code):
        raise ValueError('chat/thinking markers in code')
    ast.parse(code)  # syntax check only; never execute on host
    return code


def equivalent(actual, expected, rel=1e-6, absolute=1e-6):
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, int) and isinstance(expected, int):
        return actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isfinite(actual) and math.isfinite(expected) and math.isclose(actual, expected, rel_tol=rel, abs_tol=absolute)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(equivalent(a, b, rel, absolute) for a, b in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(equivalent(actual[k], expected[k], rel, absolute) for k in expected)
    return type(actual) is type(expected) and actual == expected


def compare(stdout, expected, fn, limits):
    if fn:
        try:
            return equivalent(json.loads(stdout), expected, limits['float_rel_tol'], limits['float_abs_tol'])
        except (ValueError, TypeError, OverflowError):
            return False
    # Strict non-whitespace tokens: no sorting, case folding or fuzzy numeric matching.
    return stdout.split() == expected.split()


from .sandbox_client import DockerValidator, docker_command

"""Runs ONLY inside Docker as uid/gid 65534. Expected answers are never supplied."""
import contextlib
import importlib.util
import json
import inspect
import sys

payload = json.loads(sys.stdin.read())
spec = importlib.util.spec_from_file_location('solution', '/tmp/task/solution.py')
module = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(sys.stderr):
    spec.loader.exec_module(module)
    try:
        target = module.Solution() if hasattr(module, 'Solution') else module
        method = getattr(target, payload['fn'])
        inspect.signature(method).bind(*payload['args'])
    except (AttributeError, TypeError, ValueError) as e:
        print('DISTILL_SIGNATURE_ERROR: ' + str(e), file=sys.stderr)
        sys.exit(2)
    answer = method(*payload['args'])
print(json.dumps(answer, allow_nan=False))

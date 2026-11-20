"""Trusted root supervisor. Forked candidate drops all privileges before exec.

No expected answers or correctness decision live in this container.
"""
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time

payload = json.loads(sys.stdin.read())
os.mkdir('/tmp/task', 0o755)
with open('/tmp/task/solution.py', 'w') as f:
    f.write(payload['code'])
os.chmod('/tmp/task/solution.py', 0o444)
limit = payload['limits']


def restrict():
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(limit['cpu_seconds']), math.ceil(limit['cpu_seconds']) + 1))
    resource.setrlimit(resource.RLIMIT_AS, (limit['memory_mb'] * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit['output_bytes'],) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)


command = ['/usr/local/bin/python', '-I', '/tmp/task/solution.py']
data = payload['input']
if payload.get('fn_name'):
    command = ['/usr/local/bin/python', '-I', '/sandbox/invoke.py']
    data = json.dumps({'fn': payload['fn_name'], 'args': data})
started = time.monotonic()
timed_out = False
# Root-owned 0600 files cannot be reopened or replaced by the candidate.
with open('/tmp/task/stdout', 'w+b') as out, open('/tmp/task/stderr', 'w+b') as err:
    os.chmod('/tmp/task/stdout', 0o600)
    os.chmod('/tmp/task/stderr', 0o600)
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err,
                            preexec_fn=restrict, cwd='/tmp', env={'PATH': '/usr/local/bin:/usr/bin',
                            'PYTHONHASHSEED': '0', 'LANG': 'C.UTF-8'})
    try:
        proc.communicate(data.encode('utf-8'), timeout=limit['wall_seconds'])
    except subprocess.TimeoutExpired:
        timed_out = True
        # The supervisor lacks CAP_KILL after child changes uid; container lifetime is the outer kill boundary.
        print(json.dumps({'status': 'timeout', 'wall_seconds': time.monotonic()-started}), flush=True)
        sys.exit(0)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    out.seek(0)
    err.seek(0)
    stdout = out.read(limit['output_bytes'] + 1)
    stderr = err.read(limit['output_bytes'] + 1)
    status = 'ok' if proc.returncode == 0 else 'runtime_error'
    if len(stdout) >= limit['output_bytes'] or len(stderr) >= limit['output_bytes']:
        status = 'output_limit'
    print(json.dumps({'status': status, 'returncode': proc.returncode,
                      'stdout': stdout.decode('utf-8', errors='replace'),
                      'stderr': stderr.decode('utf-8', errors='replace'),
                      'wall_seconds': time.monotonic()-started,
                      'cpu_seconds': usage.ru_utime + usage.ru_stime,
                      'peak_rss_kb': usage.ru_maxrss}), flush=True)

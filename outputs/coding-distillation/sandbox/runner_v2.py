"""Trusted supervisor: one container per solution, fresh unprivileged process per case."""
import json
import math
import os
import resource
import shutil
import signal
import sys
import time


def emit(value):
    print(json.dumps(value, ensure_ascii=True), flush=True)


def restrict(limits):
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(limits['cpu_seconds']), math.ceil(limits['cpu_seconds']) + 1))
    resource.setrlimit(resource.RLIMIT_AS, (limits['memory_mb'] * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits['output_bytes'],) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)


def execute(value, fn, limits):
    scratch = '/tmp/scratch'
    if os.path.exists(scratch):
        shutil.rmtree(scratch)
    os.mkdir(scratch, 0o777)
    os.chmod(scratch, 0o777)
    data = json.dumps({'fn': fn, 'args': value}) if fn else value
    with open('/tmp/task/input', 'wb') as f:
        f.write(data.encode('utf-8'))
    os.chmod('/tmp/task/input', 0o600)
    with open('/tmp/task/input', 'rb') as inp, open('/tmp/task/stdout', 'w+b') as out, open('/tmp/task/stderr', 'w+b') as err:
        os.chmod('/tmp/task/stdout', 0o600)
        os.chmod('/tmp/task/stderr', 0o600)
        started = time.monotonic()
        pid = os.fork()
        if pid == 0:
            try:
                os.dup2(inp.fileno(), 0)
                os.dup2(out.fileno(), 1)
                os.dup2(err.fileno(), 2)
                os.closerange(3, 256)
                os.chdir(scratch)
                restrict(limits)
                script = '/sandbox/invoke.py' if fn else '/tmp/task/solution.py'
                os.execve('/usr/local/bin/python', ['python', '-I', script],
                          {'PATH': '/usr/local/bin:/usr/bin', 'PYTHONHASHSEED': '0', 'LANG': 'C.UTF-8', 'TMPDIR': scratch})
            except BaseException as e:
                os.write(2, ('Candidate launch error: ' + str(e)).encode())
                os._exit(125)
        timed_out = False
        while True:
            done, status, usage = os.wait4(pid, os.WNOHANG)
            if done:
                break
            if time.monotonic() - started >= limits['wall_seconds']:
                timed_out = True
                os.kill(pid, signal.SIGKILL)
                _, status, usage = os.wait4(pid, 0)
                break
            time.sleep(0.005)
        elapsed = time.monotonic() - started
        rc = os.waitstatus_to_exitcode(status)
        out.seek(0)
        err.seek(0)
        stdout = out.read(limits['output_bytes'] + 1).decode('utf-8', errors='replace')
        stderr = err.read(limits['output_bytes'] + 1).decode('utf-8', errors='replace')
        category = 'ok' if rc == 0 else 'runtime_error'
        cpu = usage.ru_utime + usage.ru_stime
        if timed_out:
            category = 'execution_timeout'
        elif rc == -signal.SIGXCPU or cpu > limits['cpu_seconds']:
            category = 'cpu_limit'
        elif rc == -signal.SIGXFSZ or out.seek(0, 2) >= limits['output_bytes'] or err.seek(0, 2) >= limits['output_bytes']:
            category = 'output_limit'
        elif 'MemoryError' in stderr:
            category = 'memory_limit'
        elif 'SyntaxError:' in stderr or 'IndentationError:' in stderr:
            category = 'syntax_error'
        elif 'DISTILL_SIGNATURE_ERROR:' in stderr:
            category = 'incompatible_function_signature'
        elif rc == 125:
            category = 'sandbox_launch_error'
        shutil.rmtree(scratch)
        return {'status': category, 'returncode': rc, 'stdout': stdout, 'stderr': stderr,
                'wall_seconds': elapsed, 'cpu_seconds': cpu, 'peak_rss_kb': usage.ru_maxrss}


emit({'event': 'boot', 'protocol': 2})
initial = json.loads(sys.stdin.readline())
os.mkdir('/tmp/task', 0o755)
with open('/tmp/task/solution.py', 'w') as f:
    f.write(initial['code'])
os.chmod('/tmp/task/solution.py', 0o444)
emit({'event': 'ready'})
for line in sys.stdin:
    request = json.loads(line)
    if request.get('event') == 'close':
        emit({'event': 'finished'})
        break
    emit({'event': 'case_started', 'index': request['index']})
    result = execute(request['input'], initial.get('fn_name'), initial['limits'])
    emit({'event': 'case_result', 'index': request['index'], **result})

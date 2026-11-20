import ast
import json
import logging
import queue
import subprocess
import threading
import time
import uuid

from .checkers import checker_name, judge_special, support_issues, invalid_test_indices


def docker_command(name, image, limits):
    return ['docker', 'run', '--name', name, '-i', '--network', 'none', '--read-only',
            '--cap-drop', 'ALL', '--cap-add', 'SETUID', '--cap-add', 'SETGID',
            '--cap-add', 'KILL', '--cap-add', 'DAC_OVERRIDE',
            '--security-opt', 'no-new-privileges:true', '--pids-limit', str(limits['pids']),
            '--memory', f"{limits['memory_mb']}m", '--memory-swap', f"{limits['memory_mb']}m",
            '--cpus', str(limits['cpus']), '--ulimit', 'nofile=64:64',
            '--tmpfs', '/tmp:rw,noexec,nosuid,nodev,size=32m,mode=755',
            '--tmpfs', '/dev/shm:ro,noexec,nosuid,nodev,size=1m', '--log-driver', 'none', image]


class InfrastructureError(RuntimeError):
    def __init__(self, phase, message):
        self.phase = phase
        super().__init__(message)


class Session:
    def __init__(self, image, limits):
        self.limits = limits
        self.name = 'distill-v2-' + uuid.uuid4().hex
        self.start = time.monotonic()
        self.events, self.errors = [], []
        self.messages = queue.Queue()
        self.proc = subprocess.Popen(docker_command(self.name, image, limits), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                     encoding='utf-8', errors='replace', bufsize=1)
        def reader():
            try:
                for line in self.proc.stdout:
                    self.messages.put(json.loads(line))
            except Exception as e:
                self.messages.put({'event': 'protocol_error', 'error': str(e)})
            finally:
                self.messages.put({'event': 'eof'})
        def errors():
            for line in self.proc.stderr:
                self.errors.append(line)
                self.errors[:] = self.errors[-200:]
        self.reader = threading.Thread(target=reader, daemon=True)
        self.error_reader = threading.Thread(target=errors, daemon=True)
        self.reader.start()
        self.error_reader.start()

    def receive(self, expected, phase, seconds, index=None):
        try:
            message = self.messages.get(timeout=seconds)
        except queue.Empty:
            raise InfrastructureError(phase, f'No {expected} event within {seconds}s')
        self.events.append({'event': message.get('event'), 'index': message.get('index'),
                            'host_elapsed': time.monotonic()-self.start})
        if message.get('event') != expected or (index is not None and message.get('index') != index):
            raise InfrastructureError(phase, f'Unexpected protocol event: {message.get("event")}')
        return message

    def send(self, message):
        errors = []
        def write():
            try:
                self.proc.stdin.write(json.dumps(message) + '\n')
                self.proc.stdin.flush()
            except Exception as e:
                errors.append(e)
        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        writer.join(self.limits.get('transport_seconds', 30))
        if writer.is_alive() or errors:
            raise InfrastructureError('input_transport', str(errors) if errors else 'Input pipe write deadline exceeded')

    def initialize(self, code, fn):
        self.receive('boot', 'container_startup', self.limits.get('startup_seconds', 60))
        self.send({'code': code, 'fn_name': fn, 'limits': self.limits})
        self.receive('ready', 'runner_initialization', self.limits.get('transport_seconds', 30))
        self.startup_seconds = time.monotonic()-self.start

    def case(self, value, index):
        started = time.monotonic()
        self.send({'input': value, 'index': index})
        self.receive('case_started', 'case_input_transport', self.limits.get('transport_seconds', 30), index)
        result = self.receive('case_result', 'runner_execution_or_result_transport',
                              self.limits['wall_seconds'] + self.limits.get('transport_seconds', 30), index)
        result['host_case_seconds'] = time.monotonic()-started
        return result

    def close(self, failed=False):
        diagnostic = {'container': self.name, 'events': self.events, 'startup_seconds': getattr(self, 'startup_seconds', None)}
        try:
            if not failed and self.proc.poll() is None:
                self.send({'event': 'close'})
                self.receive('finished', 'runner_shutdown', self.limits.get('transport_seconds', 30))
                diagnostic['shutdown_mode'] = 'explicit_engine_remove_after_finished_ack'
                self.proc.stdin.close()
        except Exception as e:
            diagnostic['close_error'] = str(e)
        try:
            inspect = subprocess.run(['docker', 'inspect', self.name], capture_output=True, text=True, timeout=15)
            if inspect.returncode == 0:
                state = json.loads(inspect.stdout)[0]
                diagnostic['docker_state'] = state['State']
                diagnostic['docker_host_config'] = {k: state['HostConfig'].get(k) for k in
                    ('NetworkMode', 'ReadonlyRootfs', 'Memory', 'MemorySwap', 'NanoCpus', 'PidsLimit', 'CapAdd', 'CapDrop')}
            else:
                diagnostic['inspect_error'] = inspect.stderr
        except Exception as e:
            diagnostic['inspect_error'] = str(e)
        finally:
            try:
                removed = subprocess.run(['docker', 'rm', '-f', self.name], capture_output=True, text=True, timeout=20)
                diagnostic['cleanup_returncode'] = removed.returncode
                diagnostic['cleanup_stderr'] = removed.stderr
            except Exception as e:
                diagnostic['cleanup_error'] = str(e)
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait(timeout=10)
            self.reader.join(timeout=2)
            self.error_reader.join(timeout=2)
            for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                if stream and not stream.closed:
                    stream.close()
            diagnostic['docker_cli_returncode'] = self.proc.returncode
            diagnostic['docker_stderr'] = ''.join(self.errors)[-64000:]
            diagnostic['container_elapsed_seconds'] = time.monotonic()-self.start
        return diagnostic


class DockerValidator:
    def __init__(self, limits):
        self.limits = limits
        info = subprocess.run(['docker', 'info', '--format', '{{.OSType}}'], capture_output=True, text=True, timeout=20, check=True)
        if info.stdout.strip() != 'linux':
            raise RuntimeError('Docker Desktop must use Linux containers')
        image = subprocess.run(['docker', 'image', 'inspect', limits['image'], '--format', '{{.Id}}'],
                               capture_output=True, text=True, timeout=20, check=True)
        self.image = image.stdout.strip()

    def case(self, code, value, fn):
        # Convenience for isolation regression tests; same session implementation as validation.
        session = Session(self.image, self.limits)
        failed = False
        try:
            session.initialize(code, fn)
            result = session.case(value, 0)
        except InfrastructureError as e:
            failed = True
            result = {'status': 'docker_infrastructure_failure', 'phase': e.phase, 'error': str(e)}
        finally:
            diagnostics = session.close(failed)
        result['diagnostics'] = diagnostics
        return result

    def validate(self, code, problem):
        from .validate import compare
        start = time.monotonic()
        tests = problem['tests']
        result = {'accepted': False, 'tests_total': len(tests['inputs']), 'tests_run': 0,
                  'tests_passed': 0, 'cases': [], 'image_id': self.image, 'criteria': self.limits,
                  'checker': checker_name(problem), 'protocol': 2, 'infrastructure_failures': 0}
        issues = support_issues(problem)
        invalid = invalid_test_indices(problem)
        result['out_of_spec_test_indices'] = invalid
        if issues:
            result.update(status='unsupported_test_format' if 'unsupported_test_format' in issues else 'judge_review_required',
                          support_issues=issues, elapsed_seconds=time.monotonic()-start)
            return result
        try:
            ast.parse(code)
        except SyntaxError as e:
            result.update(status='syntax_error', error=str(e), elapsed_seconds=time.monotonic()-start)
            return result
        session = None
        fatal = False
        fail_fast = self.limits.get('fail_fast', True)
        problem_budget = self.limits.get('problem_seconds', 30)

        total_cases = len(tests['inputs'])
        if self.limits.get('prioritize_boundaries', True) and total_cases > 2:
            stress_idx = max(range(1, total_cases), key=lambda idx: len(str(tests['inputs'][idx])))
            order = [0, stress_idx] + [idx for idx in range(1, total_cases) if idx != stress_idx]
        else:
            order = list(range(total_cases))

        try:
            for seq_num, i in enumerate(order):
                if time.monotonic()-start > problem_budget:
                    result['status'] = 'resource_limited'
                    break
                value = tests['inputs'][i]
                expected = tests['outputs'][i]
                attempts = []
                for retry in range(self.limits.get('infrastructure_attempts', 2)):
                    try:
                        if session is None:
                            session = Session(self.image, self.limits)
                            session.initialize(code, tests['fn_name'])
                        case = session.case(value, i)
                        break
                    except (InfrastructureError, OSError) as e:
                        attempts.append({'status': 'docker_infrastructure_failure', 'phase': getattr(e, 'phase', 'launch'),
                                         'error': str(e), 'diagnostics': session.close(True) if session else {}})
                        session = None
                        result['infrastructure_failures'] += 1
                        logging.error('Sandbox infrastructure failure problem=%s case=%d: %s', problem.get('id'), i, e)
                else:
                    case = {'status': 'docker_infrastructure_failure', 'index': i}
                    fatal = True
                case['infrastructure_attempts'] = attempts
                case['passed'] = False
                if case['status'] == 'ok':
                    case['passed'] = (compare(case['stdout'], expected, tests['fn_name'], self.limits)
                                      if result['checker'] == 'provided_output' else
                                      judge_special(result['checker'], case['stdout'], value, expected))
                    if not case['passed']:
                        case['status'] = 'wrong_answer'
                        case['expected_output'] = expected
                result['cases'].append(case)
                if i in invalid:
                    case['observed_status'] = case['status']
                    case['status'] = 'out_of_spec_test'
                    case['passed'] = False
                if (seq_num+1) % 25 == 0:
                    logging.info('Validation %s: %d/%d cases', problem.get('id', '')[:10], seq_num+1, total_cases)
                if fatal:
                    break
                if fail_fast and not case['passed']:
                    result['status'] = case['status']
                    break
        finally:
            if session is not None:
                result['session_diagnostics'] = session.close()
                if (result['session_diagnostics'].get('cleanup_error') or result['session_diagnostics'].get('close_error') or
                        result['session_diagnostics'].get('cleanup_returncode') != 0):
                    result['infrastructure_failures'] += 1
        result['tests_run'] = len(result['cases'])
        result['tests_passed'] = sum(c['passed'] for c in result['cases'])
        result['accepted'] = result['tests_passed'] == result['tests_total'] and not result['infrastructure_failures']
        if result['accepted']:
            result['status'] = 'accepted'
        elif result['infrastructure_failures']:
            result['status'] = 'docker_infrastructure_failure'
        elif result.get('status') in ('resource_limited', 'validation_budget_exceeded'):
            pass
        elif any(c.get('status') == 'out_of_spec_test' for c in result['cases']):
            result['status'] = 'dataset_quality_issue'
        else:
            first_fail = next((c['status'] for c in result['cases'] if not c['passed']), 'rejected')
            result['status'] = first_fail
        result['elapsed_seconds'] = time.monotonic()-start
        return result

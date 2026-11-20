"""
Feedback-guided repair and exact function signature utilities for Configuration B.
Extracts callable signatures, embeds stubs in prompts, and formats test diagnostics for retries.
"""

import ast
from typing import Tuple, Optional


def extract_function_signature(problem: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract exact function signature and stub from problem metadata or reference AST.
    Returns (sig_str, stub_code) or (None, None) if not determinable.
    """
    fn_name = problem.get('fn_name') or problem.get('tests', {}).get('fn_name')
    if not fn_name:
        return None, None

    ref_code = problem.get('reference_solution', '')
    tree = None
    if ref_code:
        try:
            tree = ast.parse(ref_code)
        except Exception:
            pass

    args_list = []
    found = False

    if tree:
        # Priority 1: Match FunctionDef by exact fn_name
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                args_list = [a.arg for a in node.args.args]
                found = True
                break

        # Priority 2: Match Lambda assignment by fn_name (e.g. validate_ean = lambda code: ...)
        if not found:
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == fn_name and isinstance(node.value, ast.Lambda):
                            args_list = [a.arg for a in node.value.args.args]
                            found = True
                            break

        # Priority 3: Single top-level FunctionDef in reference code (aliased/wrapped)
        if not found:
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    args_list = [a.arg for a in node.args.args]
                    found = True
                    break

    # Priority 4: Fallback to input parameter count from test harness
    if not found:
        inputs = problem.get('tests', {}).get('inputs', [])
        if inputs and isinstance(inputs[0], (list, tuple)):
            args_list = [f'arg{i+1}' for i in range(len(inputs[0]))]
        else:
            args_list = ['arg1']

    params = ', '.join(args_list)
    sig_str = f"def {fn_name}({params}):"
    stub_code = f"{sig_str}\n    # Implement this function without changing its signature.\n    pass"
    return sig_str, stub_code


def _format_example_io(problem: dict) -> str:
    """Format a single public test case as an I/O example without revealing any solution code."""
    tests = problem.get('tests') or {}
    inputs = tests.get('inputs') or []
    outputs = tests.get('outputs') or []
    if not inputs or not outputs:
        return ""
    return (
        "\n\nExample input/output:\n"
        f"- Input: {_truncate(inputs[0], 160)}\n"
        f"- Expected output: {_truncate(outputs[0], 160)}"
    )


def build_initial_prompt(
    problem: dict,
    include_example: bool = False,
    chain_of_thought: bool = False
) -> Tuple[str, Optional[str]]:
    """
    Build initial prompt for Configuration B with exact function signature stub when available.
    Optional prompt-engineering additions (off by default):
      - include_example: appends one concrete input/output example from the test harness
      - chain_of_thought: asks for brief reasoning + edge-case check before a single ```python block
    Returns (prompt_text, stub_code).
    """
    base_prompt = problem['prompt']
    sig_str, stub_code = extract_function_signature(problem)

    parts = [base_prompt]
    if include_example:
        ex = _format_example_io(problem)
        if ex:
            parts.append(ex.lstrip('\n'))

    if stub_code:
        parts.append(
            "Provide your implementation using the following exact function signature:\n"
            f"```python\n{stub_code}\n```"
        )

    if chain_of_thought:
        parts.append(
            "Before writing code, briefly think step-by-step about the algorithm, edge cases "
            "(empty inputs, boundaries, exact return types), and standard-library helpers needed. "
            "Then provide your complete final implementation inside a single ```python code block."
        )

    return "\n\n".join(parts), stub_code


def _truncate(val, max_len=200):
    s = repr(val) if not isinstance(val, str) else val
    return s[:max_len] + '...' if len(s) > max_len else s


def build_failure_feedback(val_res: dict, problem: dict, stub_code: Optional[str] = None) -> str:
    """
    Construct concise, diagnostic error feedback from Docker validation results.
    Never reveals reference solution code or algorithm details.
    """
    status = val_res.get('status', 'unknown')

    if status == 'syntax_error':
        err = val_res.get('error', 'SyntaxError')
        return f"Syntax Error:\n{err}"

    if status == 'incompatible_function_signature':
        hint = f"\nExpected signature:\n```python\n{stub_code}\n```" if stub_code else ""
        return f"Function Signature Mismatch: The submitted code does not match the required function name or parameter signature.{hint}"

    cases = val_res.get('cases', [])
    failed_case = next((c for c in cases if not c.get('passed')), None)
    if not failed_case:
        return f"Validation failed with status: {status}"

    case_status = failed_case.get('status', status)
    case_idx = failed_case.get('index', 0)
    inputs = problem.get('tests', {}).get('inputs', [])
    inp = inputs[case_idx] if 0 <= case_idx < len(inputs) else 'N/A'

    if case_status == 'wrong_answer':
        expected = failed_case.get('expected_output', 'N/A')
        actual = failed_case.get('stdout', 'N/A')
        return (
            f"Test Failure (wrong_answer):\n"
            f"Test input: {_truncate(inp)}\n"
            f"Expected output: {_truncate(expected)}\n"
            f"Actual output: {_truncate(actual)}"
        )

    if case_status in ('cpu_limit', 'time_limit', 'timeout'):
        return (
            f"Test Failure (cpu_limit / timeout exceeded):\n"
            f"Test input: {_truncate(inp)}\n"
            "Execution timed out. The algorithm is too slow or entered an infinite loop on this input. Please optimize time complexity."
        )

    if case_status == 'runtime_error':
        stderr = failed_case.get('stderr', '') or failed_case.get('error', 'RuntimeError')
        lines = [ln for ln in stderr.strip().splitlines() if ln.strip()]
        short_err = '\n'.join(lines[-3:]) if lines else 'RuntimeError'
        return (
            f"Test Failure (runtime_error):\n"
            f"Test input: {_truncate(inp)}\n"
            f"Error details:\n{short_err}"
        )

    if case_status == 'memory_limit':
        return (
            f"Test Failure (memory_limit exceeded):\n"
            f"Test input: {_truncate(inp)}\n"
            "Memory budget exceeded. Reduce memory allocations and auxiliary data structures."
        )

    return f"Test Failure: status={case_status} on input={_truncate(inp)}"


def build_repair_prompt(
    problem: dict,
    previous_code: str,
    failure_feedback: str,
    stub_code: Optional[str] = None,
    attempt_num: int = 2,
    chain_of_thought: bool = False,
    prior_failures: Optional[list] = None
) -> str:
    """
    Construct a targeted repair prompt for Attempts 2..N.
    Optional prompt-engineering additions (off by default):
      - chain_of_thought: asks for brief root-cause diagnosis before a single ```python block
      - prior_failures: compact 1-line summaries of earlier failed attempts to avoid repeating mistakes
    """
    base_question = problem.get('question', problem['prompt'])
    stub_section = (
        f"\nPreserve the exact function signature:\n```python\n{stub_code}\n```"
        if stub_code else ""
    )

    history_section = ""
    if prior_failures:
        bullets = "\n".join(f"- {item}" for item in prior_failures)
        history_section = f"Summary of earlier failed attempts (do not repeat these mistakes):\n{bullets}\n\n"

    if chain_of_thought:
        closing = (
            "Briefly diagnose the root cause of the failure in 2-3 sentences, then provide "
            "the complete corrected implementation inside a single ```python code block. "
            f"Use only the Python standard library.{stub_section}"
        )
    else:
        closing = (
            "Please correct the implementation based on the test feedback above. "
            "Return only complete Python code, without Markdown fences or explanations. "
            f"Use only the Python standard library.{stub_section}"
        )

    prompt = (
        f"Problem Description:\n{base_question}\n\n"
        f"{history_section}"
        f"Attempt {attempt_num - 1} failed testing with the following diagnostic feedback:\n"
        f"----------------------------------------\n"
        f"{failure_feedback}\n"
        f"----------------------------------------\n\n"
        f"Previous code that failed:\n"
        f"```python\n{previous_code.strip()}\n```\n\n"
        f"{closing}"
    )
    return prompt

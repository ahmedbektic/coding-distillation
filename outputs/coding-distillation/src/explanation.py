import re

EXPLANATION_PROMPT_TEMPLATE = """You are an expert computer science instructor. Analyze the following Python 3 solution to the given programming problem.

Problem:
{prompt}

Python Solution:
```python
{code}
```

Provide a concise, pedagogical explanation of this solution. Your explanation must include:
1. Algorithm selection and overall approach.
2. Important steps and data structures used.
3. Why the algorithm correctly solves the problem.
4. Expected time complexity and auxiliary space complexity in Big-O notation.

Format your response clearly with these four sections. Return only the explanation without conversational preamble or internal thinking.
"""


def build_explanation_prompt(problem_text: str, code: str) -> str:
    return EXPLANATION_PROMPT_TEMPLATE.format(prompt=problem_text.strip(), code=code.strip())


def parse_complexity(text: str):
    """Extract Big-O notations for time and space complexity."""
    big_o_pattern = r'[O\u039f]\s*\(([^)]+)\)'
    matches = re.findall(big_o_pattern, text)
    time_match = re.search(r'(?:time\s*complexity|time)[\s\S]{0,80}?[O\u039f]\s*\(([^)]+)\)', text, re.IGNORECASE)
    space_match = re.search(r'(?:space\s*complexity|space|auxiliary)[\s\S]{0,80}?[O\u039f]\s*\(([^)]+)\)', text, re.IGNORECASE)

    time_comp = time_match.group(1).strip() if time_match else (matches[0].strip() if matches else None)
    space_comp = space_match.group(1).strip() if space_match else (matches[1].strip() if len(matches) > 1 else None)
    return time_comp, space_comp


def verify_explanation(explanation: str, code: str, expected_time: str = None, expected_space: str = None):
    """
    Verify that the explanation contains required pedagogical sections
    and reasonable complexity claims.
    """
    if not explanation or len(explanation.strip()) < 50:
        return {'verified': False, 'reason': 'explanation_too_short'}

    if re.search(r'<\|[^\n]*?\|>|\[/?INST\]|<</?SYS>>|</?s>|</?think>', explanation):
        return {'verified': False, 'reason': 'contains_chat_tokens'}

    text_lower = explanation.lower()

    # Check for presence of core conceptual sections
    has_approach = any(k in text_lower for k in ('approach', 'algorithm', 'method', 'strategy'))
    has_steps = any(k in text_lower for k in ('step', 'data structure', 'logic', 'implementation'))
    has_correctness = any(k in text_lower for k in ('correct', 'why', 'ensure', 'guarantee', 'handles'))
    has_complexity = any(k in text_lower for k in ('complexity', 'big-o', 'big o', 'o(', 'o ('))

    if not (has_approach and (has_steps or has_correctness) and has_complexity):
        return {'verified': False, 'reason': 'missing_required_sections',
                'details': {'approach': has_approach, 'steps': has_steps,
                            'correctness': has_correctness, 'complexity': has_complexity}}

    time_comp, space_comp = parse_complexity(explanation)
    if not time_comp and not space_comp:
        return {'verified': False, 'reason': 'no_big_o_complexity_found'}

    # Sanity checks against obvious contradictions in code
    code_lines = [l.strip() for l in code.splitlines() if l.strip() and not l.strip().startswith('#')]
    loop_count = sum(1 for l in code_lines if l.startswith(('for ', 'while ')))

    # If code has loops but claims O(1) time complexity without conditional break
    if loop_count >= 1 and time_comp and time_comp in ('1', 'O(1)'):
        # Check if the loop is constant-bounded e.g. for i in range(10)
        has_constant_range = any(re.search(r'range\s*\(\s*\d+\s*\)', l) for l in code_lines)
        if not has_constant_range:
            return {'verified': False, 'reason': 'claimed_o1_with_unbounded_loop',
                    'time_complexity': time_comp}

    # Match against dataset metadata if provided
    metadata_match = True
    if expected_time and time_comp:
        norm_expected = re.sub(r'\s+', '', expected_time.lower())
        norm_claimed = re.sub(r'\s+', '', time_comp.lower())
        if norm_expected not in norm_claimed and norm_claimed not in norm_expected:
            metadata_match = 'divergent'

    return {
        'verified': True,
        'time_complexity': time_comp,
        'space_complexity': space_comp,
        'metadata_match': metadata_match,
        'sections_present': {
            'approach': has_approach,
            'steps': has_steps,
            'correctness': has_correctness,
            'complexity': has_complexity
        }
    }

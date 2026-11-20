# Coding Sequence-Level Distillation Pipeline

A reliable, reproducible research pipeline designed to build high-quality, pure-Python coding datasets for **teacher-guided sequence-level knowledge distillation using LoRA**.

The pipeline executes on Windows 11 with an NVIDIA RTX 5060 Ti 16GB, Docker Desktop (WSL2), and locally hosted FP16 Ollama teacher models. It produces student-agnostic training examples consisting of an algorithmic problem prompt, an expert pedagogical explanation, and verified Python standard-library code.

---

## 1. Experimental Design & Distillation Configurations

The pipeline implements **exactly two primary configurations**:

### Configuration A: Reference-Solution Explanation Distillation (`run-config-a`)
- **Workflow**: For each verified problem in the shared corpus, the teacher model is provided with the problem statement and the known-correct reference solution. The teacher generates an in-depth pedagogical explanation covering:
  1. Algorithm selection and overall approach.
  2. Important steps and standard-library data structures used.
  3. Correctness justification and handling of boundary conditions.
  4. Big-O time complexity and auxiliary space complexity.
- **Verification**: The generated explanation is verified against an automated rubric ensuring all four pedagogical sections are present and complexity claims match the algorithm.
- **Output Triple**: `{problem, teacher_explanation, verified_reference_solution}`.
- **Yield**: **100% acceptance yield** (20/20 in pilot). Eliminates teacher coding errors and provides high-quality pedagogical rationales for established solutions.

### Configuration B: Independent Solution Distillation (`run-config-b`)
- **Workflow**:
  1. For callable-function problems, the exact function signature is extracted via AST and pre-populated directly into the teacher's prompt as an unalterable function stub.
  2. The teacher independently generates a Python solution.
  3. Candidate code is tested in the Docker sandbox using **fail-fast validation**.
  4. **Feedback-Guided Retries (up to 3 attempts)**:
     - **Attempt 1**: Teacher generates solution from prompt + exact signature stub.
     - **Attempt 2**: If Attempt 1 fails, the model receives its previous code and concise diagnostic feedback from the first failed test case (`wrong_answer` with input/expected/actual, `cpu_limit`, or `runtime_error`) without exposing reference code.
     - **Attempt 3**: If Attempt 2 fails, feedback from the new failure is provided for a final repair attempt.
  5. If and only if the candidate passes all test cases, the teacher generates an in-depth pedagogical explanation of its own verified code.
  6. The explanation is verified against the rubric before admitting the example to the dataset.
- **Output Triple**: `{problem, teacher_explanation, verified_teacher_solution}`.
- **Yield & Empirical Recovery**:
  - **Overall Acceptance Rate**: **60.5% (605 / 1,000 problems accepted)** on `qwen2.5:7b-instruct-fp16`.
  - **Recovery of Failed Problems**: **122 / 400 (30.5%)** previously failed zero-shot problems recovered via feedback retries.
  - **New Problems Evaluated**: **151 / 268 (56.3% pass rate)**.
  - **Attempt Distribution**: 216 solved on Attempt 1, 39 solved on Attempt 2 (Retry 1), 18 solved on Attempt 3 (Retry 2), plus 332 preserved from initial run.

---

## 2. Models & Constraints

### FP16 Ollama Teacher Models (Local Execution)
All teachers run locally via the Ollama REST API using native FP16 weights (no 4-bit or 8-bit quantization):
- `qwen2.5:7b-instruct-fp16`
- `mistral:7b-instruct-v0.3-fp16`
- `deepseek-coder:6.7b-instruct-fp16`
- `orca2:7b-fp16`
- `codellama:7b-instruct-fp16`
- `llama2:7b-chat-fp16`

Before each model runs, resident models are evicted from VRAM via `/api/generate` (`keep_alive: 0`) and the incoming teacher is preloaded and verified via `/api/show`.

### Target Student Checkpoints (Future LoRA Distillation)
The generated datasets are student-agnostic and designed for future sequence-level LoRA fine-tuning of:
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-4B`
- `tiiuae/Falcon3-1B-Instruct`
- `tiiuae/Falcon3-3B-Instruct`

*(Note: Student model training is out-of-scope for this data generation and validation pipeline.)*

### Strict Language & Source Constraints
- **Python Standard Library Only**: Every generated and reference solution must use Python 3 and its standard library only (`math`, `collections`, `itertools`, `re`, `heapq`, etc.). No external dependencies or package installations (`pip`, `numpy`, `scipy`) are permitted.
- **Strict Problem Exclusion**: All Codeforces and LeetCode problems (both direct and indirect) are **strictly excluded**.
- **Corpus Sources**:
  1. **Google Research MBPP** (official `train` split, 371 clean tasks).
  2. **TACO Codewars subset** (standard-library callable functions with clean assertions).

---

## 3. High-Performance Fail-Fast Docker Validation

The pipeline executes all candidate and reference code inside an immutable Linux container (`coding-distillation-sandbox:3`).

### Fail-Fast Optimizations
1. **Immediate Early Halting**: Validation stops immediately on the first failure (`wrong_answer`, `cpu_limit`, `runtime_error`, `incompatible_function_signature`). Solutions are only fully tested if they pass every case.
2. **Boundary & Stress Test Prioritization**: Test cases with large inputs and extreme values are evaluated first to reject slow or inefficient implementations early.
3. **Hard Budgets**:
   - Per-problem validation budget: **30 seconds total wall-clock time**.
   - Ollama generation budget: **180 seconds**.
   - Per-case CPU limit: **3.0 seconds** (with 5.0 seconds wall-clock container limit).
   - Peak RSS limit: **512 MiB**.
4. **Fine-Grained Classification**: Failures are rigorously distinguished between model errors (`wrong_answer`, `cpu_limit`, `runtime_error`) and infrastructure issues (`docker_infrastructure_failure`). Infrastructure issues are never counted as model coding failures and trigger automatic revalidation upon resumption.

### Empirical Validation Speedup Benchmarks
| Benchmark Case | Historical Validation | Fail-Fast Validation | Speedup |
|---|---|---|---|
| Pilot Problem 8 (`c08f71200e`, 100 test cases) | 705.02s (11.75 min) | **1.64s** | **430x faster** |
| Pilot Problem 15 (`5a665416eb`, 80 test cases) | 159.69s (2.66 min) | **5.09s** | **31x faster** |
| Config B 20-problem pilot run | ~90+ minutes | **4 minutes 19 seconds** | **~21x faster** |
| Infrastructure container failures | 0 | **0** | Flawless |

---

## 4. Directory Structure

```text
outputs/coding-distillation/
├── config-v3.yaml             # Primary active configuration (fail-fast, limits, repair settings)
├── Dockerfile                 # Sandbox definition (Python 3.11, non-root runner)
├── sandbox/
│   ├── Dockerfile
│   └── runner_v2.py           # Protocol v2 IPC test execution runner
├── scripts/
│   ├── resume_config_b.ps1    # Dedicated PowerShell script to resume Configuration B
│   └── run_full_experiment.ps1# Sequential execution driver for all teachers
├── src/
│   ├── corpus.py              # MBPP & Codewars candidate loader and reference corpus builder
│   ├── explanation.py         # Pedagogical explanation prompt and verification rubric
│   ├── repair.py              # AST signature extraction, stub injection, failure feedback & repair prompts
│   ├── pipeline_v2.py         # Primary CLI (prepare, run-config-a, run-config-b, report)
│   ├── report_v2.py           # Markdown & JSON summary report generator
│   ├── sandbox_client.py      # Docker validator with fail-fast & boundary prioritization
│   └── checkers.py            # AST syntax, signature compatibility, stdout normalization
├── tests/
│   ├── test_docker.py         # Isolated Docker container integration tests
│   ├── test_fail_fast.py      # Real-problem fail-fast validation benchmarks
│   ├── test_repair.py         # Unit tests for signature extraction, feedback loops, and attempt bounding
│   └── test_checkers.py       # Unit tests for checkers and signature normalization
├── data-v3/
│   ├── problems/              # Verified reference corpus (verified_corpus.jsonl)
│   ├── raw/                   # Per-problem attempt records and checkpoints
│   └── verified/              # Student-agnostic JSONL exports & provenance files
└── results-v3/
    ├── logs/                  # Detailed execution logs (pipeline.log)
    └── summaries/             # Empirical Markdown and JSON distillation reports
```

---

## 5. Windows PowerShell Execution Guide

Run all commands from `outputs/coding-distillation` in PowerShell. Use the preconfigured virtual environment at `../../work/venv`:

```powershell
# Navigate to project directory
cd outputs\coding-distillation

# 1. Verify environment, Ollama models, and Docker sandbox:
& '..\..\work\venv\Scripts\python.exe' -m src.pipeline_v2 --config config-v3.yaml doctor

# 2. Run automated test suite (including Docker sandbox integration tests):
$env:DISTILL_DOCKER_TESTS = '1'
& '..\..\work\venv\Scripts\python.exe' -m unittest discover -s tests -v
Remove-Item Env:DISTILL_DOCKER_TESTS

# 3. Build / verify the 1,000-problem Python reference corpus:
& '..\..\work\venv\Scripts\python.exe' -m src.pipeline_v2 --config config-v3.yaml prepare --target 1000

# 4. Run Configuration A across all 1,000 problems (Qwen 2.5 7B FP16):
& '..\..\work\venv\Scripts\python.exe' -m src.pipeline_v2 --config config-v3.yaml run-config-a --mode full --model qwen2.5:7b-instruct-fp16

# 5. Run Configuration B across all 1,000 problems (Exact Signatures + Feedback Retries):
& '..\..\work\venv\Scripts\python.exe' -m src.pipeline_v2 --config config-v3.yaml run-config-b --mode full --model qwen2.5:7b-instruct-fp16

# 6. Dedicated Resumption Script for Configuration B:
# (Safely skips accepted problems in 0.1ms, retries previously failed with diagnostic feedback, processes unattempted)
powershell.exe -ExecutionPolicy Bypass -File scripts\resume_config_b.ps1

# 7. Generate empirical research report and markdown summary:
& '..\..\work\venv\Scripts\python.exe' -m src.pipeline_v2 --config config-v3.yaml report --model qwen2.5:7b-instruct-fp16
```

---

## 6. Empirical Results: Full 1,000-Problem Distillation Run

Teacher model: `qwen2.5:7b-instruct-fp16` (NVIDIA RTX 5060 Ti 16GB, FP16 precision, 4096 context window).

| Metric | Configuration A (Reference Explanations) | Configuration B (Independent Solutions + Retries) |
|---|---|---|
| **Evaluated Problems** | 1,000 | 1,000 |
| **Accepted Training Examples** | **994** | **605** |
| **Acceptance Rate** | **99.4%** | **60.5%** |
| **Old Failed Problems Recovered** | N/A | **122 / 400 (30.5% recovery rate)** |
| **New Unattempted Problems Passed** | N/A | **151 / 268 (56.3% pass rate)** |
| **Solved on Attempt 1 (Stub Prompt)** | N/A | 216 problems |
| **Solved on Attempt 2 (Feedback Retry 1)**| N/A | 39 problems |
| **Solved on Attempt 3 (Feedback Retry 2)**| N/A | 18 problems |
| **Preserved From Initial Run** | N/A | 332 problems |
| **Docker Infrastructure Failures** | 0 | **0** |
| **Total Validation Wall Time** | N/A | 2,751.38s (avg 2.75s / problem) |
| **Exported Dataset File** | `data-v3/verified/qwen2.5_7b-instruct-fp16_config_a.jsonl` | `data-v3/verified/qwen2.5_7b-instruct-fp16_config_b.jsonl` |
| **Dataset Line Count (Unique)** | **994** | **605** |

---

## 7. Student-Agnostic Export Format

The pipeline exports strictly formatted JSONL datasets to `data-v3/verified/`:
- `data-v3/verified/<model>_config_a.jsonl`
- `data-v3/verified/<model>_config_b.jsonl`

Each line contains exactly two string fields:
```json
{
  "prompt": "Write a Python function to reverse the words in a given string sentence.\n\nInput: 'hello world'\nOutput: 'world hello'",
  "completion": "Algorithm selection and overall approach:\nThe solution uses Python's built-in string manipulation functions to split the string into words, reverse the order of words, and re-join them with a single space.\n\nImportant steps and data structures used:\n1. Split the string into a list of words using `s.split()`.\n2. Reverse the list using slicing `[::-1]`.\n3. Join the words using `' '.join(...)`.\n\nJustification for correctness and boundary conditions:\nHandles single-word strings, empty strings, and multiple spaces between words cleanly.\n\nTime complexity: O(N) where N is the length of the string.\nAuxiliary space complexity: O(N) to store the extracted list of words.\n\n```python\ndef reverse_words(s: str) -> str:\n    return ' '.join(s.split()[::-1])\n```\n"
}
```

Accompanying `.provenance.jsonl` files record problem IDs, original sources (MBPP / Codewars), license metadata, and teacher generation timestamps.

### Future LoRA Fine-Tuning Recipe
Because completions contain standard Markdown code fences without chat tokens or role headers, any student tokenizer can format the pairs using its native chat template during training:

```python
from datasets import load_dataset
from transformers import AutoTokenizer

student_id = "Qwen/Qwen3-1.7B"  # Or Qwen3-4B, Falcon3-1B-Instruct, Falcon3-3B-Instruct
tokenizer = AutoTokenizer.from_pretrained(student_id)
raw_dataset = load_dataset("json", data_files="data-v3/verified/qwen2.5_7b-instruct-fp16_config_a.jsonl", split="train")

def format_for_sft(row):
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "completion": [{"role": "assistant", "content": row["completion"]}]
    }

conversational = raw_dataset.map(format_for_sft, remove_columns=raw_dataset.column_names)
# Ready for Hugging Face TRL SFTTrainer
```

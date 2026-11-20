# Multi-Teacher Python Coding Distillation & Cross-Model Consensus

A reliable, reproducible research pipeline designed to evaluate 7B-class open-weight LLMs, build high-quality pure-Python coding distillation datasets, and perform cross-model consensus analysis with test-driven feedback repair.

The pipeline executes locally on Windows 11 with an NVIDIA RTX GPU (16GB VRAM), Docker Desktop (WSL2), and FP16 Ollama teacher models.

---

## 1. System Overview

The framework provides an end-to-end environment for generating student-agnostic training triples:
$$\{ \text{problem\_prompt}, \text{pedagogical\_explanation}, \text{verified\_python\_solution} \}$$

All problems and solutions strictly enforce **pure Python standard-library** implementations (0 external dependencies like `numpy` or `pip`). The corpus is frozen at 1,000 problems drawn from MBPP and Codewars (TACO callable functions).

### Primary Distillation Configurations

* **Configuration A (`run-config-a`)**: *Reference-Solution Explanation Distillation*
  * Teacher models are provided with the problem statement and the known-correct reference solution.
  * Teachers generate a structured 4-part pedagogical explanation (Approach, Data Structures, Invariant/Correctness, Big-O Complexity).
  * Explanations are validated against an automated rubric.

* **Configuration B (`run-config-b`)**: *Independent Solution Distillation*
  * The exact callable function signature is extracted via AST and pre-populated into the prompt.
  * Teachers independently author code and are validated against test suites inside isolated Docker containers.
  * Includes automated diagnostic feedback retries (up to 3 attempts) using test failure traces (`wrong_answer`, `runtime_error`, `cpu_limit`).
  * If and only if the candidate code passes all test suites, the teacher generates an explanation of its own code.

* **Cross-Model Consensus & Feedback Retries (`run-consensus-retries`)**:
  * Evaluates consensus across 6 teacher models (`Qwen2.5 7B`, `Qwen2.5-Coder 7B`, `DeepSeek-Coder 6.7B`, `CodeGemma 7B`, `Mistral 7B`, `CodeLlama 7B`).
  * Identifies high-consensus problems solved by 3, 4, or 5 models and re-evaluates the models that missed them.
  * Provides up to 5 attempts, a 250s problem budget, and targeted prompt enhancements (Chain-of-Thought reasoning, concrete input/output example, and structured retry failure history).

---

## 2. Repository Structure

```
.
├── outputs/
│   └── coding-distillation/
│       ├── config-v3.yaml             # Master pipeline configuration (models, timeouts, thresholds)
│       ├── config-v2.yaml             # Legacy configuration
│       ├── requirements.txt           # Python project dependencies
│       ├── requirements-lock.txt      # Pinned dependency lockfile
│       ├── README.md                  # Detailed distillation technical specification
│       │
│       ├── src/                       # Core pipeline source code
│       │   ├── pipeline_v2.py         # Primary pipeline engine (Config A, Config B, Consensus Retries)
│       │   ├── consensus.py           # Cross-model consensus analyzer and distribution generator
│       │   ├── generate.py            # Ollama API client with streaming and timeout controls
│       │   ├── validate.py            # Docker-isolated code validation harness
│       │   ├── repair.py              # Prompt engineering, signature injection, and feedback formatting
│       │   ├── verify.py              # Explanation rubric verification
│       │   ├── export.py              # Distillation JSONL dataset exporter
│       │   ├── report_v2.py           # Markdown/JSON report compiler
│       │   └── common.py              # File I/O utilities, schema validation, and logging helpers
│       │
│       ├── scripts/                   # Automated orchestration scripts
│       │   ├── run_6model_consensus.ps1                 # Master 6-model consensus runner
│       │   ├── run_new_coders_config_b.ps1             # Batch runner for Qwen2.5-Coder and CodeGemma
│       │   ├── finish_llama2_and_run_consensus_retries.ps1 # Targeted retry execution script
│       │   ├── check_status.py                         # Fast CLI status monitor
│       │   └── export_distill_ollama.ps1               # Dataset exporter for local LoRA distillation
│       │
│       ├── sandbox/                   # Containerized execution environment
│       │   ├── Dockerfile             # Minimal secure Python sandbox
│       │   ├── runner.py              # Container test executor with memory and CPU bounds
│       │   └── harness.py             # Sandbox test loader
│       │
│       ├── tests/                     # Unit and integration test suite
│       │   ├── test_repair.py         # Unit tests for prompt options and failure formatting
│       │   ├── test_validation.py     # Docker sandbox execution unit tests
│       │   └── test_corpus.py         # Corpus verification tests
│       │
│       └── data-v3/
│           └── problems/
│               └── verified_corpus.jsonl # Frozen 1,000-problem benchmark corpus
│
├── conversation.md                    # Research trajectory and experiment logs
├── .gitignore                         # Excludes raw traces, logs, caches, and runtime data
└── README.md                          # Root project documentation
```

---

## 3. Quickstart

### Prerequisites
* **Python 3.11+**
* **Docker Desktop** (with WSL2 backend enabled on Windows)
* **Ollama** running locally with target FP16 models pulled (e.g. `ollama pull qwen2.5-coder:7b-instruct-fp16`)

### Installation
```bash
git clone https://github.com/ahmedbektic/coding-distillation.git
cd coding-distillation

python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\Activate.ps1
pip install -r outputs/coding-distillation/requirements.txt
```

### Build Docker Sandbox
```bash
docker build -t coding-distillation-sandbox:3 outputs/coding-distillation/sandbox
```

### Running Pipeline Commands
```bash
cd outputs/coding-distillation

# 1. Run Configuration B for a specific model (1000 problems)
python -m src.pipeline_v2 --config config-v3.yaml run-config-b --mode full --model qwen2.5-coder:7b-instruct-fp16

# 2. Analyze cross-model consensus across teachers
python -m src.consensus --teachers qwen2.5:7b-instruct-fp16 qwen2.5-coder:7b-instruct-fp16 deepseek-coder:6.7b-instruct-fp16 codegemma:7b-instruct-fp16 mistral:7b-instruct-v0.3-fp16 codellama:7b-instruct-fp16

# 3. Run targeted 5-attempt consensus retries
python -m src.pipeline_v2 --config config-v3.yaml run-consensus-retries --max-attempts 5 --budget 250 --include-example --cot --retry-history
```

---

## 4. License
MIT License.

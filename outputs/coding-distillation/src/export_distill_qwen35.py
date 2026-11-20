"""
Merge a trained Qwen3.5-4B LoRA adapter into the original 16-bit checkpoint,
convert the merged checkpoint to GGUF, and generate a Qwen3.5-compatible Ollama Modelfile.

Preserves the LoRA adapter and merged 16-bit checkpoint independently even if
GGUF conversion or Ollama import fails.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# Official Qwen3.5 Ollama chat template (supports system, user, assistant, thinking toggle; defaults to non-thinking matching SFT training)
QWEN35_OLLAMA_MODELFILE_TEMPLATE = """FROM __GGUF_FILENAME__

TEMPLATE \"\"\"{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range $i, $_ := .Messages }}{{- $last := eq (len (slice $.Messages $i)) 1 }}{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if (and $last .Thinking) }}<think>
{{ .Thinking }}
</think>

{{ end }}{{ .Content }}{{ if not $last }}<|im_end|>
{{ end }}{{ end }}{{- if and $last (ne .Role "assistant") }}<|im_start|>assistant
{{ if and $.IsThinkSet $.Think }}<think>
{{ else }}<think>

</think>

{{ end }}{{ end }}{{- end }}\"\"\"

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
"""


def merge_adapter_16bit(
    adapter_dir: Path,
    merged_dir: Path,
    max_seq_length: int = 2048,
) -> dict:
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    import torch

    t0 = time.perf_counter()
    print(f"[MERGE] Loading adapter from {adapter_dir} in 16-bit BF16...", flush=True)
    model, processor = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"[MERGE] Saving merged 16-bit checkpoint to {merged_dir}...", flush=True)
    model.save_pretrained_merged(
        str(merged_dir),
        processor,
        save_method="merged_16bit",
    )

    # Ensure chat_template.jinja and processor files are present in merged_dir
    for fname in ("chat_template.jinja", "processor_config.json", "tokenizer_config.json", "tokenizer.json"):
        src = adapter_dir / fname
        dst = merged_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    elapsed = round(time.perf_counter() - t0, 2)
    safetensors = sorted(p.name for p in merged_dir.glob("*.safetensors"))
    total_bytes = sum(p.stat().st_size for p in merged_dir.iterdir() if p.is_file())
    info = {
        "merged_dir": str(merged_dir),
        "merge_seconds": elapsed,
        "safetensors_files": safetensors,
        "total_size_gb": round(total_bytes / 1024**3, 3),
        "has_chat_template": (merged_dir / "chat_template.jinja").exists(),
        "has_config": (merged_dir / "config.json").exists(),
    }
    print(f"[MERGE] Done: {json.dumps(info, indent=2)}", flush=True)
    return info


def convert_merged_to_gguf(
    merged_dir: Path,
    gguf_dir: Path,
    outtype: str = "f16",
    llama_cpp_dir: Path = Path.home() / "llama.cpp",
    ollama_compat: bool = True,
) -> dict:
    """
    Convert the merged 16-bit HF Qwen3.5-4B checkpoint to a text-only GGUF using
    llama.cpp's convert_hf_to_gguf.py (cloning llama.cpp locally if needed).

    When ollama_compat=True, disables llama.cpp HEAD's _SHARED_INPUT_MERGES
    (which fuses ssm_beta+ssm_alpha -> ssm_beta_alpha and attn_qkv+attn_gate -> attn_qkv_gate)
    so the exported GGUF preserves separate ssm_beta/ssm_alpha/attn_qkv/attn_gate tensors
    required by Ollama's bundled llama-server (v0.40.x).
    """
    t0 = time.perf_counter()
    gguf_dir.mkdir(parents=True, exist_ok=True)

    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not converter.exists():
        print(f"[GGUF] Cloning llama.cpp (shallow) to {llama_cpp_dir}...", flush=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(llama_cpp_dir)],
            check=True,
        )

    gguf_py_dir = llama_cpp_dir / "gguf-py"
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{llama_cpp_dir}:{gguf_py_dir}:{existing_pp}" if existing_pp else f"{llama_cpp_dir}:{gguf_py_dir}"

    out_file = gguf_dir / f"qwen3.5-4b-distilled-config-b-{outtype}.gguf"
    if ollama_compat:
        helper_code = (
            "import sys; "
            "import convert_hf_to_gguf as conv; "
            "import conversion.qwen as cq; "
            "cq.Qwen3_5TextModel._SHARED_INPUT_MERGES = (); "
            "sys.argv = ['convert_hf_to_gguf.py', sys.argv[1], '--outfile', sys.argv[2], '--outtype', sys.argv[3]]; "
            "conv.main()"
        )
        cmd = [sys.executable, "-c", helper_code, str(merged_dir), str(out_file), outtype]
        print(f"[GGUF] Running llama.cpp converter (ollama_compat=True, unfused ssm_beta/ssm_alpha/attn_qkv/attn_gate)...", flush=True)
    else:
        cmd = [
            sys.executable,
            str(converter),
            str(merged_dir),
            "--outfile",
            str(out_file),
            "--outtype",
            outtype,
        ]
        print(f"[GGUF] Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, env=env, check=True)

    modelfile_path = gguf_dir / "Modelfile"
    modelfile_text = QWEN35_OLLAMA_MODELFILE_TEMPLATE.replace("__GGUF_FILENAME__", out_file.name)
    modelfile_path.write_text(modelfile_text, encoding="utf-8", newline="\n")

    elapsed = round(time.perf_counter() - t0, 2)
    info = {
        "gguf_file": str(out_file),
        "outtype": outtype,
        "ollama_compat_unfused": ollama_compat,
        "gguf_size_gb": round(out_file.stat().st_size / 1024**3, 3),
        "modelfile": str(modelfile_path),
        "convert_seconds": elapsed,
    }
    print(f"[GGUF] Done: {json.dumps(info, indent=2)}", flush=True)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Qwen3.5-4B LoRA adapter and export to GGUF + Ollama Modelfile.")
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("results-v3/distillation/qwen35_4b_config_b/adapter"),
    )
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=Path("results-v3/distillation/qwen35_4b_config_b/merged_16bit"),
    )
    parser.add_argument(
        "--gguf-dir",
        type=Path,
        default=Path("results-v3/distillation/qwen35_4b_config_b/gguf"),
    )
    parser.add_argument("--outtype", default="f16", choices=["f16", "bf16", "q8_0", "f32"])
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--step", default="all", choices=["merge", "gguf", "all"])
    parser.add_argument("--llama-cpp-dir", type=Path, default=Path.home() / "llama.cpp")
    args = parser.parse_args()

    summary = {}
    if args.step in ("merge", "all"):
        summary["merge"] = merge_adapter_16bit(args.adapter_dir, args.merged_dir, args.max_seq_length)
    if args.step in ("gguf", "all"):
        summary["gguf"] = convert_merged_to_gguf(
            args.merged_dir, args.gguf_dir, outtype=args.outtype, llama_cpp_dir=args.llama_cpp_dir
        )

    report_path = args.gguf_dir.parent / "export_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[EXPORT] Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()

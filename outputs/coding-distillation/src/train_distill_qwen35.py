"""
Reusable BF16 LoRA offline sequence-level knowledge distillation training script
for Qwen/Qwen3.5-4B using Unsloth, Transformers v5, PEFT, Datasets, and TRL SFTTrainer.

Features:
- Text-only BF16 LoRA training (no 4-bit quantization); vision layers frozen & verified.
- Completion-only loss via Unsloth train_on_responses_only with explicit batch verification.
- Zero silent truncation guard: aborts if any tokenized sequence exceeds max_seq_length.
- Native Qwen3.5 chat template (non-thinking mode matching teacher explanation+code completions).
- Supports smoke tests (--max-steps), full training (--num-train-epochs), and resuming
  (--resume-from-checkpoint).
- Logs finite loss, LoRA gradient verification, token throughput, peak VRAM, duration,
  and checkpoint locations to training_report.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Unsloth MUST be imported before transformers / trl / peft for optimal patching
import unsloth  # noqa: F401
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import torch
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer


SUPPORTED_QWEN35_LANGUAGE_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

RESPONSE_PART = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION_PART = "<|im_start|>user\n"


def load_sft_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_dataset_for_qwen35(
    rows: List[Dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
) -> tuple[Dataset, Dict[str, Any]]:
    """
    Format all records using Qwen3.5's native chat template (enable_thinking=False)
    and assert that ZERO records exceed max_seq_length.
    """
    texts: List[str] = []
    full_lens: List[int] = []
    prompt_lens: List[int] = []
    completion_lens: List[int] = []
    truncated_ids: List[str] = []

    for r in rows:
        messages = r["messages"]
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if not full_text.startswith(prompt_text):
            raise ValueError(
                f"Chat template mismatch for {r.get('problem_id')}: full_text does not start with prompt_text."
            )
        if RESPONSE_PART not in full_text:
            raise ValueError(
                f"Expected Qwen3.5 response delimiter {RESPONSE_PART!r} not found in formatted text."
            )

        p_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        f_len = len(tokenizer(full_text, add_special_tokens=False)["input_ids"])
        c_len = f_len - p_len
        if f_len > max_seq_length:
            truncated_ids.append(f"{r.get('problem_id')}:{f_len}")

        texts.append(full_text)
        prompt_lens.append(p_len)
        full_lens.append(f_len)
        completion_lens.append(c_len)

    if truncated_ids:
        raise ValueError(
            f"{len(truncated_ids)} examples exceed max_seq_length={max_seq_length} "
            f"(first few: {truncated_ids[:5]}). Increase --max-seq-length to prevent truncation."
        )

    ds = Dataset.from_dict(
        {
            "problem_id": [r["problem_id"] for r in rows],
            "text": texts,
            "full_tokens": full_lens,
            "completion_tokens": completion_lens,
        }
    )

    stats = {
        "num_examples": len(rows),
        "max_seq_length": max_seq_length,
        "truncated_examples": 0,
        "min_tokens": min(full_lens),
        "median_tokens": sorted(full_lens)[len(full_lens) // 2],
        "mean_tokens": round(sum(full_lens) / len(full_lens), 2),
        "max_tokens": max(full_lens),
        "total_tokens_per_epoch": sum(full_lens),
        "total_completion_tokens_per_epoch": sum(completion_lens),
    }
    return ds, stats


def verify_completion_only_masking(trainer: SFTTrainer, tokenizer: Any) -> Dict[str, Any]:
    """
    Pull one real batch from trainer's dataloader and verify:
    - Prompt tokens are masked with -100.
    - Completion tokens + <|im_end|> are unmasked.
    """
    dl = trainer.get_train_dataloader()
    batch = next(iter(dl))
    input_ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()

    masked_ids = [tid for tid, lab in zip(input_ids, labels) if lab == -100]
    unmasked_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]

    if not masked_ids:
        raise RuntimeError("Completion-only masking failed: ZERO prompt tokens were masked (-100)!")
    if not unmasked_ids:
        raise RuntimeError("Completion-only masking failed: ALL tokens were masked (-100)!")

    # Verify masked tokens form a contiguous prefix (ignoring padding at the end if any)
    first_unmasked = next(i for i, lab in enumerate(labels) if lab != -100)
    prefix_text = tokenizer.decode(input_ids[:first_unmasked], skip_special_tokens=False)
    target_text = tokenizer.decode(unmasked_ids, skip_special_tokens=False)

    if not prefix_text.endswith(RESPONSE_PART):
        raise RuntimeError(
            f"Masked prefix did not end with expected response delimiter.\n"
            f"Prefix tail: {prefix_text[-120:]!r}\nExpected: {RESPONSE_PART!r}"
        )
    if "<|im_start|>user" in target_text or "Required function signature" in target_text:
        raise RuntimeError("Prompt text leaked into unmasked target labels!")

    return {
        "sample_total_tokens": len(input_ids),
        "sample_masked_prompt_tokens": first_unmasked,
        "sample_unmasked_completion_tokens": len(unmasked_ids),
        "masked_prefix_tail": prefix_text[-80:],
        "unmasked_target_head": target_text[:120],
        "unmasked_target_tail": target_text[-80:],
    }


class DistillationMonitorCallback(TrainerCallback):
    """Tracks step timing, finite losses, LoRA gradient flow, and GPU memory."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.step_logs: List[Dict[str, Any]] = []
        self.train_start_time: Optional[float] = None
        self.last_step_time: Optional[float] = None
        self.lora_grad_checked = False
        self.lora_grad_summary: Dict[str, Any] = {}
        self.total_tokens_seen = 0

    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        now = time.perf_counter()
        self.train_start_time = now
        self.last_step_time = now

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if not self.lora_grad_checked:
            lora_with_grad = 0
            lora_nonzero_grad = 0
            lora_finite = True
            vision_with_grad = 0
            max_grad = 0.0
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if "visual" in name or "vision" in name:
                        vision_with_grad += 1
                    if "lora_" in name:
                        lora_with_grad += 1
                        g = param.grad.detach()
                        if not torch.isfinite(g).all():
                            lora_finite = False
                        gn = g.float().abs().max().item()
                        if gn > 0.0:
                            lora_nonzero_grad += 1
                        if gn > max_grad:
                            max_grad = gn
            self.lora_grad_checked = True
            self.lora_grad_summary = {
                "checked_at_step": state.global_step + 1,
                "lora_params_with_grad": lora_with_grad,
                "lora_params_with_nonzero_grad": lora_nonzero_grad,
                "all_lora_grads_finite": lora_finite,
                "max_abs_lora_grad": round(max_grad, 6),
                "vision_params_with_grad": vision_with_grad,
            }
            print(f"[GRAD CHECK] {json.dumps(self.lora_grad_summary)}", flush=True)
            if lora_nonzero_grad == 0 or not lora_finite or vision_with_grad > 0:
                raise RuntimeError(f"Gradient verification failed: {self.lora_grad_summary}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        now = time.perf_counter()
        dt = now - (self.last_step_time or now)
        self.last_step_time = now
        loss = logs.get("loss")
        if loss is not None and not math.isfinite(float(loss)):
            raise RuntimeError(f"Non-finite training loss detected at step {state.global_step}: {loss}")
        entry = {
            "step": state.global_step,
            "epoch": round(float(logs.get("epoch", state.epoch or 0.0)), 4),
            "loss": round(float(loss), 6) if loss is not None else None,
            "grad_norm": round(float(logs["grad_norm"]), 6) if "grad_norm" in logs and logs["grad_norm"] is not None else None,
            "learning_rate": logs.get("learning_rate"),
            "step_wall_seconds": round(dt, 3),
            "elapsed_seconds": round(now - (self.train_start_time or now), 2),
            "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
            "vram_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "vram_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        }
        if "loss" in logs:
            self.step_logs.append(entry)


def main() -> None:
    parser = argparse.ArgumentParser(description="BF16 LoRA SFT distillation for Qwen/Qwen3.5-4B.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data-v3/distillation/qwen2.5_7b_config_b_sft.jsonl"),
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-adapter-dir", type=Path, default=None)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optim", default="adamw_torch_fused")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument(
        "--limit-examples",
        type=int,
        default=0,
        help="Optional subset size (0 = all eligible examples).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for training, but torch.cuda.is_available() is False.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    gpu_props = torch.cuda.get_device_properties(0)
    print(
        f"[ENV] GPU={gpu_props.name} CC={gpu_props.major}.{gpu_props.minor} "
        f"VRAM={gpu_props.total_memory / 1024**3:.2f}GB BF16={torch.cuda.is_bf16_supported()}",
        flush=True,
    )

    t0_load = time.perf_counter()
    model, processor = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        use_exact_model_name=True,
        use_gradient_checkpointing="unsloth",
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Explicitly ensure every vision parameter is frozen before attaching LoRA
    for name, param in model.named_parameters():
        if "visual" in name or "vision" in name:
            param.requires_grad = False

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=SUPPORTED_QWEN35_LANGUAGE_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )

    vision_trainable = [
        n for n, p in model.named_parameters() if p.requires_grad and ("visual" in n or "vision" in n)
    ]
    if vision_trainable:
        raise RuntimeError(f"Vision parameters unexpectedly trainable: {vision_trainable[:5]}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    load_seconds = round(time.perf_counter() - t0_load, 2)
    print(
        f"[MODEL] Loaded {args.model_name} in {load_seconds}s | "
        f"total={total_params:,} trainable={trainable_params:,} ({100*trainable_params/total_params:.3f}%) | "
        f"vision_trainable=0 | base_vram_alloc={torch.cuda.memory_allocated()/1024**3:.2f}GB",
        flush=True,
    )

    rows = load_sft_jsonl(args.dataset)
    if args.limit_examples > 0:
        rows = rows[: args.limit_examples]

    train_dataset, ds_stats = format_dataset_for_qwen35(rows, tokenizer, args.max_seq_length)
    print(f"[DATASET] {json.dumps(ds_stats)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)
    planned_total_steps = (
        args.max_steps
        if args.max_steps > 0
        else math.ceil(steps_per_epoch * args.num_train_epochs)
    )
    warmup_steps = max(1, int(round(planned_total_steps * args.warmup_ratio)))

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        optim=args.optim,
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=4,
        report_to="none",
        dataset_num_proc=1,
        remove_unused_columns=True,
    )

    monitor = DistillationMonitorCallback(model)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=sft_config,
        callbacks=[monitor],
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )

    mask_check = verify_completion_only_masking(trainer, tokenizer)
    print(f"[MASK CHECK] {json.dumps(mask_check, ensure_ascii=False)}", flush=True)

    t0_train = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_wall_seconds = round(time.perf_counter() - t0_train, 2)

    peak_alloc_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    peak_reserved_gb = round(torch.cuda.max_memory_reserved() / 1024**3, 3)

    # Compute token throughput across processed examples
    actual_steps = trainer.state.global_step
    if args.max_steps > 0:
        approx_examples_seen = min(len(train_dataset) * args.num_train_epochs, actual_steps * effective_batch_size)
        avg_tokens = ds_stats["mean_tokens"]
        approx_tokens_seen = int(round(approx_examples_seen * avg_tokens))
    else:
        approx_tokens_seen = int(round(ds_stats["total_tokens_per_epoch"] * args.num_train_epochs))

    tokens_per_sec = round(approx_tokens_seen / max(train_wall_seconds, 1e-6), 2)

    # Save final adapter separately if requested
    saved_adapter_dir = None
    if args.save_adapter_dir is not None:
        args.save_adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(args.save_adapter_dir))
        processor.save_pretrained(str(args.save_adapter_dir))
        saved_adapter_dir = str(args.save_adapter_dir)
        print(f"[SAVE] Saved LoRA adapter + processor to {saved_adapter_dir}", flush=True)

    checkpoints = sorted(
        [str(p) for p in args.output_dir.glob("checkpoint-*") if p.is_dir()]
    )

    # Steady-state step timing (exclude step 1 kernel compilation overhead when estimating)
    steady_steps = [s["step_wall_seconds"] for s in monitor.step_logs if s["step"] >= 3]
    mean_steady_step_s = round(sum(steady_steps) / len(steady_steps), 3) if steady_steps else None
    full_2epoch_steps = math.ceil(steps_per_epoch * 2)
    est_full_2epoch_seconds = (
        round(
            (monitor.step_logs[0]["step_wall_seconds"] if monitor.step_logs else 0.0)
            + (full_2epoch_steps - 1) * mean_steady_step_s,
            1,
        )
        if mean_steady_step_s is not None
        else None
    )

    report = {
        "model_name": args.model_name,
        "dataset_path": str(args.dataset),
        "dataset_stats": ds_stats,
        "hardware": {
            "gpu": gpu_props.name,
            "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
            "total_vram_gb": round(gpu_props.total_memory / 1024**3, 3),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
        "training_config": {
            "precision": "bf16_lora_no_4bit",
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": 0.0,
            "target_modules": SUPPORTED_QWEN35_LANGUAGE_MODULES,
            "vision_frozen": True,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_pct": round(100.0 * trainable_params / total_params, 4),
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": "cosine",
            "warmup_steps": warmup_steps,
            "optim": args.optim,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "gradient_checkpointing": "unsloth",
            "max_seq_length": args.max_seq_length,
            "num_train_epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "completion_only_loss": True,
            "chat_template_enable_thinking": False,
            "save_steps": args.save_steps,
        },
        "mask_verification": mask_check,
        "gradient_verification": monitor.lora_grad_summary,
        "results": {
            "global_steps": actual_steps,
            "train_loss": round(float(train_result.training_loss), 6),
            "initial_step_loss": monitor.step_logs[0]["loss"] if monitor.step_logs else None,
            "final_step_loss": monitor.step_logs[-1]["loss"] if monitor.step_logs else None,
            "model_load_seconds": load_seconds,
            "train_wall_seconds": train_wall_seconds,
            "train_wall_minutes": round(train_wall_seconds / 60.0, 2),
            "approx_tokens_processed": approx_tokens_seen,
            "approx_tokens_per_second": tokens_per_sec,
            "mean_steady_step_seconds": mean_steady_step_s,
            "estimated_full_2epoch_steps": full_2epoch_steps,
            "estimated_full_2epoch_seconds": est_full_2epoch_seconds,
            "estimated_full_2epoch_minutes": (
                round(est_full_2epoch_seconds / 60.0, 2) if est_full_2epoch_seconds is not None else None
            ),
            "peak_vram_allocated_gb": peak_alloc_gb,
            "peak_vram_reserved_gb": peak_reserved_gb,
            "checkpoints": checkpoints,
            "saved_adapter_dir": saved_adapter_dir,
        },
        "step_logs": monitor.step_logs,
    }

    report_path = args.output_dir / "training_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== TRAINING SUMMARY ===", flush=True)
    print(json.dumps({k: v for k, v in report.items() if k != "step_logs"}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

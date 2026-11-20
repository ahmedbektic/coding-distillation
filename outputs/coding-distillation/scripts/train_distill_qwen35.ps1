param(
    [switch]$Smoke,
    [string]$ResumeFromCheckpoint = "",
    [int]$Epochs = 2,
    [int]$MaxSeqLength = 2048,
    [int]$LoraR = 16,
    [int]$LoraAlpha = 16,
    [double]$LearningRate = 0.0001,
    [int]$SaveSteps = 25,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

# Ensure no Ollama model is resident in GPU VRAM before training
$loaded = ollama ps 2>$null
if ($loaded -and ($loaded | Measure-Object -Line).Lines -gt 1) {
    Write-Host "WARNING: Ollama currently has models loaded in VRAM:" -ForegroundColor Yellow
    $loaded | Write-Host
}

$wslProjectDir = "/mnt/c/Users/bektic/Documents/Codex/2026-11-13/build-a-complete-runnable-python-research/outputs/coding-distillation"

if ($Smoke) {
    $outDir = "results-v3/distillation/smoke"
    $extraArgs = "--max-steps 8 --save-steps 4"
} else {
    $outDir = "results-v3/distillation/qwen35_4b_config_b/checkpoints"
    $extraArgs = "--num-train-epochs $Epochs --max-steps -1 --save-steps $SaveSteps --save-adapter-dir results-v3/distillation/qwen35_4b_config_b/adapter"
}

if ($ResumeFromCheckpoint -ne "") {
    $extraArgs = "$extraArgs --resume-from-checkpoint $ResumeFromCheckpoint"
}

$cmd = "set -e; source ~/venvs/unsloth-qwen35/bin/activate; cd $wslProjectDir; python -m src.train_distill_qwen35 --dataset data-v3/distillation/qwen2.5_7b_config_b_sft.jsonl --model-name Qwen/Qwen3.5-4B --output-dir $outDir --max-seq-length $MaxSeqLength --lora-r $LoraR --lora-alpha $LoraAlpha --learning-rate $LearningRate --per-device-train-batch-size 1 --gradient-accumulation-steps 8 --logging-steps 1 --seed $Seed $extraArgs"

Write-Host "Running WSL2 Unsloth training command..." -ForegroundColor Cyan
wsl -d Ubuntu -- bash -lc $cmd

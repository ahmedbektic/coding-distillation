param(
    [ValidateSet("merge", "gguf", "all")]
    [string]$Step = "all",
    [ValidateSet("f16", "bf16", "q8_0", "f32")]
    [string]$OutType = "f16",
    [string]$OllamaModelName = "test-model-distilled"
)

$ErrorActionPreference = "Stop"

$wslProjectDir = "/mnt/c/Users/bektic/Documents/Codex/2026-11-13/build-a-complete-runnable-python-research/outputs/coding-distillation"
$cmd = "set -e; source ~/venvs/unsloth-qwen35/bin/activate; cd $wslProjectDir; python -m src.export_distill_qwen35 --adapter-dir results-v3/distillation/qwen35_4b_config_b/adapter --merged-dir results-v3/distillation/qwen35_4b_config_b/merged_16bit --gguf-dir results-v3/distillation/qwen35_4b_config_b/gguf --outtype $OutType --step $Step"

Write-Host "Exporting merged 16-bit checkpoint and GGUF ($OutType)..." -ForegroundColor Cyan
wsl -d Ubuntu -- bash -lc $cmd

if ($Step -eq "gguf" -or $Step -eq "all") {
    $ggufDir = Join-Path $PSScriptRoot "..\results-v3\distillation\qwen35_4b_config_b\gguf"
    Write-Host "Importing GGUF into Ollama as '$OllamaModelName'..." -ForegroundColor Cyan
    Push-Location $ggufDir
    try {
        ollama create $OllamaModelName -f Modelfile
    } finally {
        Pop-Location
    }
    ollama show $OllamaModelName
}

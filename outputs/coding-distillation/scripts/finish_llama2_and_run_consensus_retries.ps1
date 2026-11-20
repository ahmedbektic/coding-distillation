# Master workflow script:
# 1. Finishes remaining problems for llama2:7b-chat-fp16 (resumes from problem 659 to 1000)
# 2. Computes cross-model consensus across all 5 active 7B FP16 models
# 3. Executes targeted 5-attempt feedback retries on problems solved by 3 or 4 models
# 4. Updates all verified datasets and compiles final research summaries

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

$Python = "..\..\work\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "Virtual environment python not found at $Python"
    exit 1
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "PHASE 1: Finishing Remaining Problems for LLaMA-2 7B" -ForegroundColor Cyan
Write-Host "Model: llama2:7b-chat-fp16 (Resuming from problem 659)" -ForegroundColor Cyan
Write-Host "Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$Phase1Start = [System.Diagnostics.Stopwatch]::StartNew()

try {
    & $Python -m src.pipeline_v2 --config config-v3.yaml run-config-b --mode full --model llama2:7b-chat-fp16
    Write-Host "`nUpdating report for llama2:7b-chat-fp16..." -ForegroundColor Gray
    & $Python -m src.pipeline_v2 --config config-v3.yaml report --model llama2:7b-chat-fp16
}
catch {
    Write-Host "Error occurred during LLaMA-2 run: $_" -ForegroundColor Red
}
finally {
    # Unload model from VRAM
    try {
        ollama stop llama2:7b-chat-fp16 | Out-Null
    } catch {}
}

$Phase1Start.Stop()
Write-Host "`nPhase 1 Completed in $($Phase1Start.Elapsed.TotalMinutes.ToString('F1')) minutes." -ForegroundColor Green

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "PHASE 2: Computing Baseline Consensus (Problems solved by 0..5 models)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

& $Python -m src.pipeline_v2 --config config-v3.yaml consensus

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "PHASE 3: Targeted 5-Attempt Retries on High-Consensus Problems" -ForegroundColor Cyan
Write-Host "Criteria: Re-ask problems solved by 3 or 4 models with 5 attempts" -ForegroundColor Cyan
Write-Host "Parameters: max_attempts=5, problem_budget=250s, timeout=75s" -ForegroundColor Cyan
Write-Host "Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$Phase3Start = [System.Diagnostics.Stopwatch]::StartNew()

try {
    & $Python -m src.pipeline_v2 --config config-v3.yaml run-consensus-retries --max-attempts 5 --budget 250
}
catch {
    Write-Host "Error occurred during consensus retries: $_" -ForegroundColor Red
}

$Phase3Start.Stop()
Write-Host "`nPhase 3 Completed in $($Phase3Start.Elapsed.TotalMinutes.ToString('F1')) minutes." -ForegroundColor Green

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "ALL PHASES COMPLETED!" -ForegroundColor Green
Write-Host "End Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

# Script to resume Configuration B across remaining and previously failed problems
# Using exact function signature stubs, feedback-guided retries (up to 3 attempts),
# 180s per-problem total budget, and automatic explanation verification.

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
Write-Host "Resuming Configuration B (Exact Signatures + Feedback Retries)" -ForegroundColor Cyan
Write-Host "Model: qwen2.5:7b-instruct-fp16" -ForegroundColor Cyan
Write-Host "Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Step 1: Execute Configuration B across all 1,000 problems
# (skips accepted in milliseconds, retries failures with feedback, processes unattempted)
Write-Host "`nExecuting Configuration B..." -ForegroundColor Yellow
& $Python -m src.pipeline_v2 --config config-v3.yaml run-config-b --mode full --model qwen2.5:7b-instruct-fp16

# Step 2: Compile empirical distillation report
Write-Host "`nCompiling empirical distillation report..." -ForegroundColor Yellow
& $Python -m src.pipeline_v2 --config config-v3.yaml report --model qwen2.5:7b-instruct-fp16

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "Configuration B Distillation Pipeline Completed!" -ForegroundColor Green
Write-Host "End Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

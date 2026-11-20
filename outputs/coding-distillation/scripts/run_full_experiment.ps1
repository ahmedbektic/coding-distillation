# Runner script for full 1,000-problem sequence distillation experiment
# Executes Configuration A (Reference Explanations), Configuration B (Independent Solutions),
# and compiles the final research report.

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
Write-Host "Starting Full 1,000-Problem Distillation Pipeline" -ForegroundColor Cyan
Write-Host "Model: qwen2.5:7b-instruct-fp16" -ForegroundColor Cyan
Write-Host "Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Step 1: Preflight Doctor Check
Write-Host "`n[1/4] Running environment and sandbox preflight checks..." -ForegroundColor Yellow
& $Python -m src.pipeline_v2 --config config-v3.yaml doctor

# Step 2: Configuration A (Reference Explanations across 1,000 problems)
Write-Host "`n[2/4] Executing Configuration A (Reference-Solution Explanation Distillation)..." -ForegroundColor Yellow
& $Python -m src.pipeline_v2 --config config-v3.yaml run-config-a --mode full --model qwen2.5:7b-instruct-fp16

# Step 3: Configuration B (Independent Solution Generation & Fail-Fast Docker Validation)
Write-Host "`n[3/4] Executing Configuration B (Independent Solution Distillation)..." -ForegroundColor Yellow
& $Python -m src.pipeline_v2 --config config-v3.yaml run-config-b --mode full --model qwen2.5:7b-instruct-fp16

# Step 4: Final Empirical Distillation Report Generation
Write-Host "`n[4/4] Compiling empirical distillation reports..." -ForegroundColor Yellow
& $Python -m src.pipeline_v2 --config config-v3.yaml report --model qwen2.5:7b-instruct-fp16

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "Full 1,000-Problem Distillation Pipeline Completed!" -ForegroundColor Green
Write-Host "End Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

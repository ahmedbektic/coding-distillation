# Runner script for Configuration B across the two new 7B FP16 coding teacher models:
# 1. qwen2.5-coder:7b-instruct-fp16
# 2. codegemma:7b-instruct-fp16
#
# Standard Configuration B rules (NOT consensus retries):
# - max_attempts: 3 (Initial stub prompt + up to 2 feedback-guided retries)
# - problem_budget_seconds: 150s
# - ollama.timeout_seconds: 75s
# - Fail-fast Docker validation (stops on 1st error)
# - Strict resumption: skips previously evaluated problems on restart

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

$Python = "..\..\work\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "Virtual environment python not found at $Python"
    exit 1
}

$Teachers = @(
    "codegemma:7b-instruct-fp16"
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Starting Configuration B for Qwen2.5-Coder & CodeGemma FP16" -ForegroundColor Cyan
Write-Host "Teachers: $($Teachers -join ', ')" -ForegroundColor Cyan
Write-Host "Parameters: max_attempts=3, problem_budget=150s, timeout=75s" -ForegroundColor Cyan
Write-Host "Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$GlobalStart = [System.Diagnostics.Stopwatch]::StartNew()

for ($i = 0; $i -lt $Teachers.Count; $i++) {
    $teacher = $Teachers[$i]
    $step = $i + 1
    Write-Host "`n[$step/$($Teachers.Count)] >>> Running Configuration B for $teacher <<<" -ForegroundColor Yellow
    $TeacherStart = [System.Diagnostics.Stopwatch]::StartNew()

    try {
        & $Python -m src.pipeline_v2 --config config-v3.yaml run-config-b --mode full --model $teacher
        Write-Host "Updating cumulative empirical distillation report for $teacher..." -ForegroundColor Gray
        & $Python -m src.pipeline_v2 --config config-v3.yaml report --model $teacher
    }
    catch {
        Write-Host "Error occurred during run for $($teacher): $_" -ForegroundColor Red
    }
    finally {
        try {
            ollama stop $teacher | Out-Null
        } catch {}
    }

    $TeacherStart.Stop()
    Write-Host "Finished $teacher in $($TeacherStart.Elapsed.TotalMinutes.ToString('F1')) minutes." -ForegroundColor Green
}

$GlobalStart.Stop()
Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "Both New Coding Teachers Completed in $($GlobalStart.Elapsed.TotalHours.ToString('F2')) hours!" -ForegroundColor Green
Write-Host "End Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

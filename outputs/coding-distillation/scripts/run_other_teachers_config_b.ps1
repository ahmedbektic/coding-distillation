# Runner script for Configuration B across the remaining 2 LLaMA FP16 teacher models:
# 1. codellama:7b-instruct-fp16
# 2. llama2:7b-chat-fp16
#
# (Completed teachers: qwen2.5:7b-instruct-fp16, mistral:7b-instruct-v0.3-fp16, deepseek-coder:6.7b-instruct-fp16)
# (Excluded teacher: orca2:7b-fp16 due to high format extraction error rate / 3.28% yield)
#
# Configured with:
# - max_attempts: 3 (Initial attempt + 2 feedback retries)
# - problem_budget_seconds: 150s
# - ollama.timeout_seconds: 75s
# - Fail-fast Docker validation (stops on 1st error)
# - AST function signature stub injection
#
# Progress is checkpointed per-problem to disk in data-v3/raw/<teacher>/config_b/

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
    "codellama:7b-instruct-fp16",
    "llama2:7b-chat-fp16"
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Starting Configuration B for Remaining 2 LLaMA Teachers" -ForegroundColor Cyan
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
        # Continue to next teacher or allow manual inspection
    }
    finally {
        # Unload model from VRAM to ensure clean switch for next model
        try {
            curl.exe -s http://localhost:11434/api/generate -d "{`"model`": `"$teacher`", `"keep_alive`": 0}" | Out-Null
        } catch {}
    }

    $TeacherStart.Stop()
    Write-Host "Finished $teacher in $($TeacherStart.Elapsed.TotalMinutes.ToString('F1')) minutes." -ForegroundColor Green
}

$GlobalStart.Stop()
Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "Both LLaMA Teachers Completed in $($GlobalStart.Elapsed.TotalHours.ToString('F2')) hours!" -ForegroundColor Green
Write-Host "End Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

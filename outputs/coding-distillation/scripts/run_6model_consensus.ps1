# Master Orchestration Script:
# 1. Monitors / waits for codegemma:7b-instruct-fp16 to reach 1000 problems in Config B
# 2. Computes the baseline cross-model consensus across all 6 models:
#    - qwen2.5:7b-instruct-fp16
#    - qwen2.5-coder:7b-instruct-fp16
#    - deepseek-coder:6.7b-instruct-fp16
#    - codegemma:7b-instruct-fp16
#    - mistral:7b-instruct-v0.3-fp16
#    - codellama:7b-instruct-fp16
# 3. Executes targeted 5-attempt feedback retries on problems solved by high consensus (3, 4, or 5 models)
#    Parameters: max_attempts=5, problem_budget=250s, timeout=75s/100s
#    Prompt additions: --include-example, --cot, --retry-history
# 4. Updates all verified datasets (data-v3/verified/*_config_b.jsonl) and compiles final summaries

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
    "qwen2.5:7b-instruct-fp16",
    "qwen2.5-coder:7b-instruct-fp16",
    "deepseek-coder:6.7b-instruct-fp16",
    "codegemma:7b-instruct-fp16",
    "mistral:7b-instruct-v0.3-fp16",
    "codellama:7b-instruct-fp16"
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "6-MODEL CONSENSUS PIPELINE" -ForegroundColor Cyan
Write-Host "Teachers: $($Teachers -join ', ')" -ForegroundColor Cyan
Write-Host "Target Consensus Levels: Solved by 3, 4, or 5 models" -ForegroundColor Cyan
Write-Host "Retry Configuration: max_attempts=5, budget=250s, prompt enhancements (example, CoT, failure history)" -ForegroundColor Cyan
Write-Host "Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# PHASE 1: Verify / wait for CodeGemma to reach 1000 evaluated problems
Write-Host "`n>>> Checking CodeGemma 7B FP16 completion status..." -ForegroundColor Yellow
$GemmaRawDir = "data-v3/raw/codegemma_7b-instruct-fp16/config_b"

function Get-GemmaCount {
    if (-not (Test-Path $GemmaRawDir)) { return 0 }
    return (Get-ChildItem -Path $GemmaRawDir -Filter "record.json" -Recurse).Count
}

$currentCount = Get-GemmaCount
while ($currentCount -lt 1000) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] CodeGemma currently at $currentCount/1000 problems. Waiting for background run..." -ForegroundColor Gray
    Start-Sleep -Seconds 60
    $newCount = Get-GemmaCount
    if ($newCount -eq $currentCount) {
        # Check if the pipeline process is still running or if we need to resume it
        $lockFile = "data-v3/.pipeline.lock"
        if (-not (Test-Path $lockFile)) {
            Write-Host "No active pipeline process detected. Resuming CodeGemma Config B to reach 1000..." -ForegroundColor Yellow
            try {
                & $Python -m src.pipeline_v2 --config config-v3.yaml run-config-b --mode full --model codegemma:7b-instruct-fp16
            } catch {
                Write-Host "Resumption encountered error: $_" -ForegroundColor Red
            }
        }
    }
    $currentCount = Get-GemmaCount
}

Write-Host "`nCodeGemma 7B FP16 has completed all 1,000 problems!" -ForegroundColor Green

# Ensure Gemma is unloaded from VRAM
try {
    ollama stop codegemma:7b-instruct-fp16 | Out-Null
} catch {}

# PHASE 2: Generate baseline report and compute baseline consensus
Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "PHASE 2: Computing Baseline 6-Model Consensus (0..6 Solves)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

& $Python -m src.pipeline_v2 --config config-v3.yaml report --model codegemma:7b-instruct-fp16
& $Python -m src.consensus --teachers $Teachers --consensus-levels 3 4 5

# Copy baseline consensus for reference comparison
$ResultsDir = "results-v3/summaries"
if (Test-Path "$ResultsDir/consensus_summary.json") {
    Copy-Item "$ResultsDir/consensus_summary.json" "$ResultsDir/consensus_summary_baseline.json" -Force
}
if (Test-Path "$ResultsDir/consensus_summary.md") {
    Copy-Item "$ResultsDir/consensus_summary.md" "$ResultsDir/consensus_summary_baseline.md" -Force
}

# PHASE 3: Targeted 5-Attempt Consensus Retries
Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "PHASE 3: Running Targeted 5-Attempt Consensus Retries" -ForegroundColor Cyan
Write-Host "Criteria: Re-evaluate problems solved by 3, 4, or 5 models for any teacher that missed them" -ForegroundColor Cyan
Write-Host "Enhancements: 1 concrete example, Chain-of-Thought reasoning, failure history" -ForegroundColor Cyan
Write-Host "Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$Phase3Start = [System.Diagnostics.Stopwatch]::StartNew()

try {
    & $Python -m src.pipeline_v2 --config config-v3.yaml run-consensus-retries `
        --max-attempts 5 `
        --budget 250 `
        --include-example `
        --cot `
        --retry-history `
        --teachers $Teachers `
        --consensus-levels 3 4 5
}
catch {
    Write-Host "Error occurred during consensus retries: $_" -ForegroundColor Red
}
finally {
    # Unload any remaining model from VRAM
    foreach ($t in $Teachers) {
        try { ollama stop $t | Out-Null } catch {}
    }
}

$Phase3Start.Stop()
Write-Host "`nConsensus Retries Completed in $($Phase3Start.Elapsed.TotalMinutes.ToString('F1')) minutes." -ForegroundColor Green

# PHASE 4: Final Summary & Verification
Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "ALL PHASES COMPLETE — UPDATED 6-MODEL CONSENSUS REPORT" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

& $Python -m src.consensus --teachers $Teachers --consensus-levels 3 4 5
Write-Host "`nExecution Finished at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green

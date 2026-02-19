# Suggest a commit message for your staged changes (or current repo).
# Run from any repo: .\suggest.ps1   or   .\suggest.ps1 C:\path\to\repo
# Requires: trained checkpoint at $PROJECT_ROOT\runs\best.pt

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$RepoPath = if ($args.Count -gt 0) { $args[0] } else { Get-Location }.Path

$env:PYTHONPATH = "$ProjectRoot;$ProjectRoot\packages"
$Checkpoint = Join-Path $ProjectRoot "runs\best.pt"
if (-not (Test-Path $Checkpoint)) {
    Write-Host "Checkpoint not found. Train first: python -m src.train --config configs/default.yaml"
    exit 1
}
& python -m src.generate --repo $RepoPath --staged --checkpoint $Checkpoint --intent

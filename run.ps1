# Run the Commit Message Generator project
# Usage: .\run.ps1 [prepare|train|generate|eval]

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$env:PYTHONPATH = "$ProjectRoot;$ProjectRoot\packages"

function Run-Prepare {
    python -m src.prepare_data
}

function Run-Train {
    python -m src.train --config configs/default.yaml
}

function Run-Generate {
    param([string]$Diff = "data/sample.diff", [switch]$Intent)
    $args = @("--diff", $Diff, "--checkpoint", "runs/best.pt")
    if ($Intent) { $args += "--intent" }
    python -m src.generate @args
}

function Run-Eval {
    python -m src.evaluate --checkpoint runs/best.pt --data data/val.jsonl
}

$cmd = if ($args.Count -gt 0) { $args[0] } else { "generate" }
switch ($cmd) {
    "prepare" { Run-Prepare }
    "train"   { Run-Train }
    "generate" { Run-Generate }
    "eval"    { Run-Eval }
    "all"     { Run-Prepare; Run-Train; Run-Generate }
    default   { Write-Host "Usage: .\run.ps1 [prepare|train|generate|eval|all]"; exit 1 }
}

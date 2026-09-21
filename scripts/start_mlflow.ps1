$projectRoot = Split-Path -Parent $PSScriptRoot
$artifactRoot = Join-Path $projectRoot "mlartifacts"

New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null

Push-Location $projectRoot
try {
    & ".\.venv\Scripts\mlflow.exe" server `
        --backend-store-uri "sqlite:///mlflow.db" `
        --artifacts-destination "./mlartifacts" `
        --host "127.0.0.1" `
        --port 5000
}
finally {
    Pop-Location
}

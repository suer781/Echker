$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root
& "$Root\.venv-life\Scripts\python.exe" "$Root\run_agent.py" --config "$Root\config.json" --state-dir "$Root\state" $args
exit $LASTEXITCODE

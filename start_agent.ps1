$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root
& "$Root\.venv-life\Scripts\python.exe" "$Root\run_agent.py" --state-dir "$Root\state_qwen_dual" $args
exit $LASTEXITCODE

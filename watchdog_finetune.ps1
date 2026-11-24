$ErrorActionPreference = "Stop"
Write-Host "The legacy ByT5 fine-tuning watchdog is disabled on the Qwen mainline."
Write-Host "Use .\start_agent.ps1 for online Qwen learning; no fine-tuning job was started."
exit 1

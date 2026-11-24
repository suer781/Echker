$ErrorActionPreference = "Stop"
Write-Host "The legacy byte-level dataset watchdog is disabled on the Qwen mainline."
Write-Host "Run .\start_agent.ps1 for online Qwen learning; no dataset job was started."
exit 1

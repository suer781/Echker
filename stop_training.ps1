# stop_training.ps1 — 停机维护（你主动停，不是看门狗误杀）
# 行为：创建 watchdog.stop，看门狗在 ≤30 秒内终止当前训练并退出。
# 代价：损失自上次进度提交以来的 ≤16 帧训练，恢复时自动重放，不丢数据。
$stateDir = "O:\Documents\智能自学习AI（新）\state_dataset"
New-Item -ItemType File -Path (Join-Path $stateDir "watchdog.stop") -Force | Out-Null
Write-Host "watchdog.stop 已创建。"
Write-Host "看门狗将在 30 秒内停止训练并退出；用 status_check.ps1 确认已停。"
Write-Host "维护完成后运行 start_training.ps1 恢复。"

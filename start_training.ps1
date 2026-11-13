# start_training.ps1 — 停机维护后恢复训练（需要你手动运行，不是自启动）
# 行为：删除 watchdog.stop → 检查没有别的研究会话在写同一检查点 → WMI 启动看门狗。
$root     = "O:\Documents\智能自学习AI（新）"
$stateDir = Join-Path $root "state_dataset"
$stop     = Join-Path $stateDir "watchdog.stop"

# 0) 已有看门狗在岗？直接退出，避免双看门狗
$pidFile = Join-Path $stateDir "watchdog.pid"
if (Test-Path -LiteralPath $pidFile) {
    $wdPid = (Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue) -as [int]
    if ($wdPid -and (Get-Process -Id $wdPid -ErrorAction SilentlyContinue)) {
        Write-Host "看门狗已在运行 (pid=$wdPid)，无需启动。如需重启请先 stop_training.ps1。"
        exit 0
    }
}

# 1) 已有训练 python 在跑？拒绝双写检查点
$launchers = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$root*" }
if ($launchers) {
    Write-Host "检测到训练 python 仍在运行（pid=$($launchers.Id -join ', ')），拒绝启动第二个写者。"
    Write-Host "如确认无占用，请先 stop_training.ps1 并等 status_check.ps1 显示已停。"
    exit 1
}

# 2) 清掉历史 stop 文件
if (Test-Path -LiteralPath $stop) { Remove-Item -LiteralPath $stop -Force }

# 3) WMI 分离式启动看门狗（不随终端/会话结束）
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "O:\Documents\智能自学习AI（新）\watchdog.ps1"'
}
if ($r.ReturnValue -eq 0) {
    Write-Host "看门狗已启动 PID=$($r.ProcessId)。30 秒后用 status_check.ps1 验证训练起来。"
} else {
    Write-Host "启动失败 ReturnValue=$($r.ReturnValue)"
    exit 1
}

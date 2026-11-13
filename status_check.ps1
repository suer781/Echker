# status_check.ps1 — 一键查看全量在线学习状态
# 用法：powershell -NoProfile -ExecutionPolicy Bypass -File status_check.ps1
$ErrorActionPreference = "Continue"
$root     = "O:\Documents\智能自学习AI（新）"
$stateDir = Join-Path $root "state_dataset"

Write-Host "=== 进程 ==="
# venv 的 python.exe 是启动器桩，真正的解释器是它的子进程，内存/CPU 要看子进程
$launchers = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$root*" }
if ($launchers) {
    foreach ($p in $launchers) {
        Write-Host ("launcher pid={0}  RSS={1:N0}MB  启动={2}" -f $p.Id, ($p.WorkingSet64/1MB), $p.StartTime)
        $kids = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)" -ErrorAction SilentlyContinue
        foreach ($k in $kids) {
            $kp = Get-Process -Id $k.ProcessId -ErrorAction SilentlyContinue
            if ($kp) {
                Write-Host ("  worker pid={0} ({1})  RSS={2:N0}MB  CPU总={3:N0}s" -f $kp.Id, $k.Name, ($kp.WorkingSet64/1MB), $kp.TotalProcessorTime.TotalSeconds)
            }
        }
    }
} else { Write-Host "python 未在运行" }

$wdPid = $null
$pidFile = Join-Path $stateDir "watchdog.pid"
if (Test-Path -LiteralPath $pidFile) { $wdPid = (Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue) -as [int] }
$wd = if ($wdPid) { Get-Process -Id $wdPid -ErrorAction SilentlyContinue } else { $null }
Write-Host ("watchdog: {0}" -f $(if ($wd) { "运行中 pid=$wdPid" } else { "未运行" }))

Write-Host "`n=== 数据集进度 ==="
$prog = Get-Content -LiteralPath (Join-Path $stateDir "dataset_progress.json") -Raw -Encoding UTF8 | ConvertFrom-Json
$doneFiles = @($prog.files.PSObject.Properties | Where-Object { $_.Value.complete }).Count
$allFiles  = @($prog.files.PSObject.Properties).Count
Write-Host ("帧数: {0}   完成文件: {1}/{2}" -f $prog.frames, $doneFiles, $allFiles)
$cur = $prog.files.PSObject.Properties | Sort-Object { $_.Value.next_offset } | Select-Object -Last 1
if ($cur) { Write-Host ("当前: {0}  offset={1}/{2}" -f $cur.Name, $cur.Value.next_offset, $cur.Value.size) }

Write-Host "`n=== 最新一次输出 ==="
$newest = Get-ChildItem -LiteralPath $stateDir -Filter "stdout_*.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($newest) {
    Write-Host ("文件: {0}   最后写入: {1}" -f $newest.Name, $newest.LastWriteTime)
    Get-Content -LiteralPath $newest.FullName -Tail 2 -Encoding UTF8
} else { Write-Host "尚无 stdout 日志" }

Write-Host "`n=== 检查点 ==="
foreach ($f in @("latest.pt", "latest.backup.pt")) {
    $item = Get-Item -LiteralPath (Join-Path $stateDir $f) -ErrorAction SilentlyContinue
    if ($item) { Write-Host ("{0}  {1:N1}MB  {2}" -f $f, ($item.Length/1MB), $item.LastWriteTime) }
}

Write-Host "`n=== watchdog 最近日志 ==="
$watchlog = Join-Path $stateDir "watchdog.log"
if (Test-Path -LiteralPath $watchlog) { Get-Content -LiteralPath $watchlog -Tail 6 -Encoding UTF8 } else { Write-Host "无日志" }

# watchdog.ps1 — 守护 ingest_dataset.py 全量在线学习
# 功能：崩溃自动重启 / 停滞检测（stdout 无输出 + CPU 冻结才杀）/ RSS 记录 / 优雅完成检测
# 停止方式：在 state_dataset 下创建 watchdog.stop 文件（空文件即可）
# 日志：state_dataset\watchdog.log；每次运行的 stdout/stderr 独立成文件
param(
    [int]$StallMinutes = 20,
    [int]$MaxRestarts = 200,
    [int]$PollSeconds = 30
)
$ErrorActionPreference = "Continue"

$root     = "O:\Documents\智能自学习AI（新）"
$stateDir = Join-Path $root "state_dataset"
$stopFile = Join-Path $stateDir "watchdog.stop"
$py       = Join-Path $root ".venv-life\Scripts\python.exe"
$ingest   = Join-Path $root "ingest_dataset.py"
$watchlog = Join-Path $stateDir "watchdog.log"
$pidFile  = Join-Path $stateDir "watchdog.pid"

function Write-Log([string]$msg) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -LiteralPath $watchlog -Value $line -Encoding UTF8
}

Set-Location -LiteralPath $root
$PID | Out-File -LiteralPath $pidFile -Encoding ascii
Write-Log "watchdog started pid=$PID stall=${StallMinutes}min maxRestarts=$MaxRestarts poll=${PollSeconds}s"

function Get-RunIsComplete {
    # 仅当刚写完 last_run.json 且退出码为 0 时视为优雅完成
    $lastRun = Join-Path $stateDir "last_run.json"
    if (Test-Path -LiteralPath $lastRun) {
        $age = ((Get-Date) - (Get-Item -LiteralPath $lastRun).LastWriteTime).TotalSeconds
        return ($age -lt 300)
    }
    return $false
}

function Get-TreeProcs([int]$rootPid) {
    # venv python.exe 是启动器桩；真实解释器在子进程里，内存/CPU 必须按进程树合计
    $result = @()
    $p = Get-Process -Id $rootPid -ErrorAction SilentlyContinue
    if ($p) { $result += $p }
    $kids = Get-CimInstance Win32_Process -Filter "ParentProcessId=$rootPid" -ErrorAction SilentlyContinue
    foreach ($k in $kids) {
        $kp = Get-Process -Id $k.ProcessId -ErrorAction SilentlyContinue
        if ($kp) { $result += $kp }
    }
    return $result
}

function Get-TreeRssMb([object[]]$procs) {
    $sum = 0
    foreach ($p in $procs) { $sum += $p.WorkingSet64 }
    return ($sum / 1MB)
}

function Get-TreeCpuSeconds([object[]]$procs) {
    $sum = 0.0
    foreach ($p in $procs) { $sum += $p.TotalProcessorTime.TotalSeconds }
    return $sum
}

$restarts = 0
while ($true) {
    if (Test-Path -LiteralPath $stopFile) { Write-Log "stop file present; exiting"; break }

    $stamp  = Get-Date -Format "yyyyMMdd_HHmmss"
    $outLog = Join-Path $stateDir ("stdout_{0}.log" -f $stamp)
    $errLog = Join-Path $stateDir ("stderr_{0}.log" -f $stamp)

    $proc = Start-Process -FilePath $py `
        -ArgumentList @("-u", "-X", "utf8", $ingest, "--root", "O:\数据集", "--train", "--state-dir", $stateDir, "--save-every", "16") `
        -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog
    if (-not $proc) { Write-Log "failed to start python; watchdog exits"; break }
    Write-Log "started run#$restarts pid=$($proc.Id) stdout=$(Split-Path -Leaf $outLog)"

    $lastOutputTick = Get-Date
    $lastMemLog     = Get-Date
    $stoppedByFile  = $false

    while ($true) {
        Start-Sleep -Seconds $PollSeconds

        if (Test-Path -LiteralPath $stopFile) {
            Write-Log "stop file present during run; killing pid=$($proc.Id)"
            & taskkill /PID $proc.Id /T /F 2>$null | Out-Null
            $stoppedByFile = $true
            break
        }
        if ($proc.HasExited) { break }

        $out = Get-Item -LiteralPath $outLog -ErrorAction SilentlyContinue
        if ($out -and $out.LastWriteTime -gt $lastOutputTick) { $lastOutputTick = Get-Date }

        if (((Get-Date) - $lastMemLog).TotalSeconds -ge 300) {
            $tree = Get-TreeProcs $proc.Id
            if ($tree) { Write-Log ("pid={0} tree_rss={1:N0}MB ({2} procs)" -f $proc.Id, (Get-TreeRssMb $tree), $tree.Count) }
            $lastMemLog = Get-Date
        }

        if (((Get-Date) - $lastOutputTick).TotalMinutes -lt $StallMinutes) { continue }

        # 超过停滞窗口：用 60 秒 CPU 增量区分“活着但慢/系统休眠刚唤醒”和“真挂死”
        $tree1 = Get-TreeProcs $proc.Id
        if (-not $tree1) { break }
        $cpu1 = Get-TreeCpuSeconds $tree1
        Start-Sleep -Seconds 60
        $tree2 = Get-TreeProcs $proc.Id
        if (-not $tree2) { break }
        $cpuDelta = (Get-TreeCpuSeconds $tree2) - $cpu1
        $out = Get-Item -LiteralPath $outLog -ErrorAction SilentlyContinue
        if ($cpuDelta -lt 2 -and (-not ($out -and $out.LastWriteTime -gt $lastOutputTick))) {
            Write-Log ("stall detected: no stdout >{0}min and cpu delta {1:N1}s/60s; killing pid={2}" -f $StallMinutes, $cpuDelta, $proc.Id)
            & taskkill /PID $proc.Id /T /F 2>$null | Out-Null
            Start-Sleep -Seconds 5
            break
        }
        Write-Log ("slow-but-alive: cpu delta {0:N1}s/60s; reset stall window" -f $cpuDelta)
        $lastOutputTick = Get-Date
    }

    $exitCode = 0
    try { $exitCode = $proc.ExitCode } catch { $exitCode = -9999 }
    Write-Log "run#$restarts exited code=$exitCode"

    if ($stoppedByFile) { Write-Log "stopped by request; watchdog exits"; break }
    if ($exitCode -eq 0 -and (Get-RunIsComplete)) {
        Write-Log "graceful completion detected (dataset fully consumed); watchdog exits"
        break
    }

    $restarts += 1
    if ($restarts -ge $MaxRestarts) { Write-Log "max restarts reached; watchdog exits"; break }
    Start-Sleep -Seconds 15
}
Write-Log "watchdog finished"

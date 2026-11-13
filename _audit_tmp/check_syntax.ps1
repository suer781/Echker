$root = "O:\Documents\智能自学习AI（新）"
foreach ($f in @("watchdog.ps1", "status_check.ps1", "stop_training.ps1", "start_training.ps1")) {
    $errs = $null
    $null = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $root $f), [ref]$null, [ref]$errs)
    if ($errs.Count -eq 0) {
        Write-Host "$f OK"
    } else {
        foreach ($e in $errs) {
            Write-Host "$f ERROR: $($e.Message) @ line $($e.Extent.StartLineNumber)"
        }
    }
}

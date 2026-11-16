# 看门狗：训练中断自动重启（周期保存使续跑不丢进度）
$root = "O:\Documents\智能自学习AI（新）"
$log = "$root\finetune_fresh.log"
$wlog = "$root\finetune_watch.log"
while ($true) {
  $done = Select-String -Path $log -Pattern '"done"' -Quiet -ErrorAction SilentlyContinue
  if ($done) { Add-Content $wlog "$(Get-Date) 训练完成"; break }
  $running = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$root\.venv-life*" }
  if (-not $running) {
    Add-Content $wlog "$(Get-Date) 训练中断，自动重启"
    Start-Process -FilePath "$root\.venv-life\Scripts\python.exe" -ArgumentList "-u","finetune_byt5.py","--resume","--epochs","1","--lr","6e-5","--data-dir","finetune_data_r2" -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError "$root\finetune_fresh.err"
  }
  Start-Sleep -Seconds 120
}

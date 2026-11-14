$os = Get-CimInstance Win32_OperatingSystem
Write-Host ("物理内存: {0:N1}GB  可用: {1:N1}GB" -f ($os.TotalVisibleMemorySize/1MB), ($os.FreePhysicalMemory/1MB))

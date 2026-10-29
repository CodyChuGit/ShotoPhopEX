# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Restart the portable ComfyUI with the given extra flags, wait until it answers, print its argv.
# Python runs unbuffered (-u) so the log file fills as it goes.
param([string]$Flags = "")
$root = "C:\Users\5090\ComfyUI\ComfyUI_windows_portable"
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*ComfyUI\main.py*" } | ForEach-Object {
    "stopping pid $($_.ProcessId)"
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 3
$argList = "-u -s `"$root\ComfyUI\main.py`" --windows-standalone-build --listen 127.0.0.1 --port 8188 --preview-method auto $Flags"
$log = "C:\Users\5090\ComfyUI\comfyui-$((Get-Date).ToString('yyyyMMdd-HHmmss')).log"
$p = Start-Process -FilePath "$root\python_embeded\python.exe" -ArgumentList $argList -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru
"started pid $($p.Id), log $log"
Set-Content -Path "C:\Users\5090\ComfyUI\current-log.txt" -Value $log -Encoding ascii
$deadline = (Get-Date).AddSeconds(180)
while ((Get-Date) -lt $deadline) {
    try {
        $s = Invoke-RestMethod -Uri http://127.0.0.1:8188/system_stats -TimeoutSec 3
        "up: ComfyUI $($s.system.comfyui_version), argv: $($s.system.argv -join ' ')"
        exit 0
    } catch { Start-Sleep -Seconds 2 }
}
"server did not come up in 180 s; log tail:"
Get-Content $log -Tail 20
Get-Content "$log.err" -Tail 20
exit 1

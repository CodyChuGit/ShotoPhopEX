# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Measure a ComfyUI launch flag on a generative Spot Healing dab: starts the portable server
# with the measured flags plus -Flags, waits for the port, runs same-graph-probe.ps1 (one
# candidate twice, then Best of two), prints the server's staging ("Model Initialization")
# and run times, stops the server. Log in $Log.
param(
    [string]$Flags = "",
    [string]$Log = ""
)
$root = "C:\Users\5090\ComfyUI\ComfyUI_windows_portable"
$tag = if ($Flags) { ($Flags -replace "[^A-Za-z0-9]+", "-").Trim("-") } else { "baseline" }
if (-not $Log) { $Log = "C:\Users\5090\ComfyUI\photocraft-tests\bench\flag-$tag.log" }
Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*ComfyUI_windows_portable*" } | ForEach-Object { Stop-Process -Id $_.Id -Force -Confirm:$false }
Start-Sleep -Seconds 2
$launch = @("-u", "-s", "$root\ComfyUI\main.py", "--windows-standalone-build", "--listen", "127.0.0.1", "--port", "8188", "--preview-method", "auto", "--fast", "fp8_matrix_mult") + ($Flags -split "\s+" | Where-Object { $_ })
$server = Start-Process -FilePath "$root\python_embeded\python.exe" -ArgumentList $launch -WorkingDirectory $root -RedirectStandardOutput "$Log.out" -RedirectStandardError "$Log.err" -PassThru -WindowStyle Hidden
$deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $deadline -and -not (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 2 }
if (-not (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue)) { "server not up ($tag)"; Get-Content "$Log.err" -Tail 20; exit 1 }
"=== flags: '$Flags'"
powershell -NoProfile -File "$PSScriptRoot\same-graph-probe.ps1" 2>&1 | ForEach-Object { "$_" }
Stop-Process -Id $server.Id -Force -Confirm:$false
Start-Sleep -Seconds 2
Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*ComfyUI_windows_portable*" } | ForEach-Object { Stop-Process -Id $_.Id -Force -Confirm:$false }
"--- server log"
Get-Content "$Log.out", "$Log.err" | ForEach-Object { $_ -replace "\x1b\[[0-9;]*m", "" } |
    Where-Object { $_ -match "Initialization complete|Prompt executed|got prompt|patches attached|partially|lowvram|Error|error" } |
    ForEach-Object { (($_ -replace "\s+", " ") -replace "Model (\w+) prepared for dynamic VRAM loading\. (\d+)MB Staged\. (\d+) patches attached.*", 'staged $1 $2MB $3 patches') -replace "^.*?(\d+%\|.*?\| \d+/\d+ \[[\d:]+<[^,]*, [^,]*, )", '$1' } |
    ForEach-Object { $_.Substring(0, [Math]::Min(100, $_.Length)) }

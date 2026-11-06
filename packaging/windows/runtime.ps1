<#
.SYNOPSIS
  Assemble the native engine's runtime folder (plan-native-engine §10, Phase 5).

.DESCRIPTION
  A build with the native engine (`--features native-diffusion`) delay-loads NVIDIA's CUDA
  libraries; none of them ship inside photocraft.exe. This script copies the files the
  `--engine` worker needs from a CUDA toolkit's bin\x64 into one folder (default: dist\runtime),
  writes THIRD_PARTY.txt naming each component and its licence, and prints the size. It
  downloads nothing.

  The app finds the folder next to photocraft.exe (`runtime\`) when the preference
  Integrations › Runtime folders is empty; the portable zip of package.ps1 -Native carries it.

  SAM 3.1 runs on candle from its checkpoint, so these libraries serve every feature; there is
  no ONNX Runtime or cuDNN (both removed on 2026-10-29).

  Licences (read them before publishing a build; upstream.md has the checklist):
    CUDA runtime, cuBLAS, cuBLASLt, cuRAND   NVIDIA CUDA Toolkit EULA, Attachment A (redistributable)

.EXAMPLE
  pwsh packaging/windows/runtime.ps1 -Cuda C:\Users\5090\scoop\apps\cuda\current\bin\x64
#>
param(
  [Parameter(Mandatory)] [string] $Cuda,
  [string] $Out
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $Out) { $Out = Join-Path $Root 'dist\runtime' }

# What the worker loads: candle's cuBLAS / cuBLASLt / cuRAND and the CUDA runtime (delay-loaded
# by the apps' build scripts).
$Components = @(
  @{ Name = 'CUDA runtime libraries'; Licence = 'NVIDIA CUDA Toolkit EULA, Attachment A'; Dir = $Cuda
     Files = 'cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll', 'curand64_10.dll' }
)

foreach ($c in $Components) {
  foreach ($f in $c.Files) {
    $p = Join-Path $c.Dir $f
    if (-not (Test-Path $p)) { throw "$($c.Name): $f is not in $($c.Dir)" }
  }
}

Remove-Item -Recurse -Force $Out -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$notice = @(
  'Third-party libraries in this folder, loaded by PhotoCraft''s native engine (photocraft.exe --engine).',
  'Each is redistributed under its own licence; none is covered by PhotoCraft''s MIT OR Apache-2.0 licence.',
  ''
)
$total = 0
foreach ($c in $Components) {
  $size = 0
  foreach ($f in $c.Files) {
    $src = Join-Path $c.Dir $f
    Copy-Item $src $Out
    $size += (Get-Item $src).Length
  }
  $total += $size
  $notice += "$($c.Name): $($c.Licence)"
  $notice += '  ' + ($c.Files -join ', ')
  $notice += ''
  Write-Output ('{0,-32} {1,8:N0} MB' -f $c.Name, ($size / 1MB))
}
Set-Content -Path (Join-Path $Out 'THIRD_PARTY.txt') -Value $notice -Encoding utf8
Write-Output ('{0,-32} {1,8:N0} MB  -> {2}' -f 'runtime folder', ($total / 1MB), $Out)

<#
.SYNOPSIS
  Assemble the native engine's runtime folder (plan-native-engine §10, Phase 5).

.DESCRIPTION
  A build with the native engine (`--features native-diffusion`) delay-loads the CUDA libraries
  and loads ONNX Runtime and cuDNN at run time; none of them ship inside photocraft.exe. This
  script copies the files the `--engine` worker needs from folders already on this PC into one
  folder (default: dist\runtime), writes THIRD_PARTY.txt naming each component and its licence,
  and prints the size. It downloads nothing: point it at a CUDA toolkit's bin\x64, an ONNX
  Runtime GPU build's lib\ and a cuDNN 9 bin folder.

  The app finds the folder next to photocraft.exe (`runtime\`) when the preference
  Integrations › Runtime folders is empty; the portable zip of package.ps1 -Native carries it.

  -Ort and -Cudnn are optional since the Rust rebuild (2026-10-27): SAM 3.1 runs on candle from
  its checkpoint, so a folder without them (NVIDIA's CUDA libraries only, about 700 MB less)
  serves every feature when the models come from SAM 3.1's checkpoint; ONNX Runtime and cuDNN
  are only for the exported SAM graphs. Give both or neither.

  Licences (read them before publishing a build; upstream.md has the checklist):
    CUDA runtime, cuBLAS, cuBLASLt, cuRAND   NVIDIA CUDA Toolkit EULA, Attachment A (redistributable)
    cuDNN 9                                  NVIDIA cuDNN Software License Agreement (redistributable)
    ONNX Runtime                             MIT

.EXAMPLE
  pwsh packaging/windows/runtime.ps1 -Cuda C:\Users\5090\scoop\apps\cuda\current\bin\x64 `
    -Ort C:\Users\5090\Documents\_eval\onnxruntime-cuda13\onnxruntime-win-x64-gpu_cuda13-1.31.0\lib `
    -Cudnn C:\path\to\cudnn\bin
#>
param(
  [Parameter(Mandatory)] [string] $Cuda,
  [string] $Ort,
  [string] $Cudnn,
  [string] $Out
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $Out) { $Out = Join-Path $Root 'dist\runtime' }
if ([bool]$Ort -ne [bool]$Cudnn) { throw 'give both -Ort and -Cudnn (the exported SAM graphs) or neither (SAM 3.1 on candle)' }

# What the worker loads. CUDA: candle's cuBLAS / cuBLASLt / cuRAND (delay-loaded by the apps'
# build scripts) and the runtime ONNX Runtime's CUDA provider imports. cuDNN: the provider opens
# cudnn64_9.dll, which opens its sub-libraries by name; tensor_ir is the one ONNX Runtime 1.31's
# preload list predates (gn-perception preloads it).
$Components = @(
  @{ Name = 'CUDA runtime libraries'; Licence = 'NVIDIA CUDA Toolkit EULA, Attachment A'; Dir = $Cuda
     Files = 'cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll', 'curand64_10.dll' },
  @{ Name = 'ONNX Runtime (CUDA provider)'; Licence = 'MIT'; Dir = $Ort
     Files = 'onnxruntime.dll', 'onnxruntime_providers_shared.dll', 'onnxruntime_providers_cuda.dll' },
  @{ Name = 'cuDNN 9'; Licence = 'NVIDIA cuDNN Software License Agreement'; Dir = $Cudnn
     Files = 'cudnn64_9.dll', 'cudnn_graph64_9.dll', 'cudnn_ops64_9.dll', 'cudnn_cnn64_9.dll', 'cudnn_adv64_9.dll',
             'cudnn_heuristic64_9.dll', 'cudnn_engines_precompiled64_9.dll', 'cudnn_engines_runtime_compiled64_9.dll',
             'cudnn_engines_tensor_ir64_9.dll', 'cudnn_ext64_9.dll' }
) | Where-Object { $_.Dir }

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

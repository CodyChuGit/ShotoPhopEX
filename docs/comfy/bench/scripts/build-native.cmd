@rem Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
@echo off
rem PhotoshopEX with the diffusion engine compiled in (docs/comfy/dev-environment-windows.md).
rem %1 = package (photocraft-cli | photocraft); %2 = CARGO_TARGET_DIR (optional)
set PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set NVCC_APPEND_FLAGS=-Xcompiler /Zc:preprocessor -Xcompiler /MD -std=c++17
set CUDARC_CUDA_VERSION=13030
set CUTLASS_DIR=C:\Users\5090\Documents\_eval\cutlass\include
set CUDA_COMPUTE_CAP=120
set PATH=C:\Users\5090\scoop\persist\rustup\.cargo\bin;C:\Users\5090\scoop\apps\cuda\current\bin;%PATH%
if not "%2"=="" set CARGO_TARGET_DIR=%2
cd /d "%~dp0..\..\..\.."
cargo build --release -p %1 --features native-diffusion

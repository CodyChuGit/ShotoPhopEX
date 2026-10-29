@rem Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
@echo off
rem Runs a command with the native engine's build environment (MSVC, nvcc flags, CUTLASS, cargo,
rem CUDA on PATH), in the current folder: e.g. from C:\Users\5090\Documents\genai-native
rem   C:\Users\5090\Documents\PhotoshopEX\docs\comfy\bench\scripts\native-env.cmd cargo test --release -p qir
rem build-native.cmd is the same environment for PhotoshopEX's two packages.
set PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set NVCC_APPEND_FLAGS=-Xcompiler /Zc:preprocessor -Xcompiler /MD -std=c++17
set CUDARC_CUDA_VERSION=13030
set CUTLASS_DIR=C:\Users\5090\Documents\_eval\cutlass\include
set CUDA_COMPUTE_CAP=120
set PATH=C:\Users\5090\scoop\persist\rustup\.cargo\bin;C:\Users\5090\scoop\apps\cuda\current\bin;C:\Users\5090\scoop\apps\cuda\current\bin\x64;%PATH%
%*

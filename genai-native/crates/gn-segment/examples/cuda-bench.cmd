@echo off
set PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set NVCC_APPEND_FLAGS=-Xcompiler /Zc:preprocessor -Xcompiler /MD -std=c++17
set CUDARC_CUDA_VERSION=13030
set CUDA_COMPUTE_CAP=120
set CUTLASS_DIR=C:\Users\5090\Documents\_eval\cutlass\include
set PATH=C:\Users\5090\scoop\persist\rustup\.cargo\bin;C:\Users\5090\scoop\apps\cuda\current\bin;%PATH%
set CARGO_TARGET_DIR=%CD%\crates\gn-segment\target-cuda
cargo %*

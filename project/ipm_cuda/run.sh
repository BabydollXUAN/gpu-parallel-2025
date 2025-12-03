#!/usr/bin/env bash
set -e

echo "[*] Compiling ipm_cuda.cu -> ipm_cuda"
nvcc -O2 -std=c++14 -o ipm_cuda ipm_cuda.cu

echo "[*] Running ipm_cuda"
./ipm_cuda

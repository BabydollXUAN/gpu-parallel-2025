#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage: bash run.sh <N>"
  exit 1
fi

N=$1

echo "[*] Compiling lab4_cuda_fft.cu -> lab4_cuda_fft"
nvcc -O2 -arch=sm_80 -lcufft lab4_cuda_fft.cu -o lab4_cuda_fft

echo "[*] Running lab4_cuda_fft with N=${N}"
./lab4_cuda_fft ${N}

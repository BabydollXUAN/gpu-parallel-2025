#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage: bash run.sh <N> [BLOCK_SIZE]"
  echo "Example: bash run.sh 2048 16"
  exit 1
fi

N=$1
BLOCK_SIZE=${2:-16}

echo "[*] Compiling lab5_cuda_image.cu -> lab5_cuda_image"
nvcc -O2 -std=c++14 lab5_cuda_image.cu -o lab5_cuda_image

echo "[*] Running lab5_cuda_image with N=${N}, BLOCK_SIZE=${BLOCK_SIZE}"
./lab5_cuda_image ${N} ${BLOCK_SIZE}

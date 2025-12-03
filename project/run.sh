#!/usr/bin/env bash
set -e

BUILD_DIR=build
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# 用最简单的方式：直接 nvcc 编译 main.cu（先不折腾 CMake）
echo "[*] Compiling project main.cu"
nvcc -O2 -o project_demo ../src/main.cu

echo "[*] Running project_demo"
./project_demo

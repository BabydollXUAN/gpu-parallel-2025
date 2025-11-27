#!/usr/bin/env bash
set -e

SRC=lab3_openmp_matmul.cpp
BIN=lab3_openmp_matmul

echo "[*] Compiling ${SRC} -> ${BIN}"
g++ -O2 -fopenmp -o ${BIN} ${SRC}

echo "[*] Running ${BIN}"
./${BIN} "$@"

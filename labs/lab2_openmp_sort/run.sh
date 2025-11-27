#!/usr/bin/env bash
set -e

SRC=lab2_openmp_sort.cpp
BIN=lab2_openmp_sort

echo "[*] Compiling ${SRC} -> ${BIN}"
g++ -O2 -fopenmp -o ${BIN} ${SRC}

echo "[*] Running ${BIN}"
./${BIN} "$@"

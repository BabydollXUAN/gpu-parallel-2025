set -e
SRC=lab1_vec_add.cu
BIN=lab1_vec_add

echo "[*] Compiling ${SRC} -> ${BIN}"
nvcc -O2 -o ${BIN} ${SRC}

echo "[*] Running ${BIN}"
./${BIN}

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <omp.h>

// 访问 A(i, j)：行优先存储
inline double get(const std::vector<double>& M, int n, int i, int j) {
    return M[i * n + j];
}

inline void set(std::vector<double>& M, int n, int i, int j, double v) {
    M[i * n + j] = v;
}

// 串行矩阵乘：C = A * B
void matmul_serial(const std::vector<double>& A,
                   const std::vector<double>& B,
                   std::vector<double>& C,
                   int n)
{
    std::fill(C.begin(), C.end(), 0.0);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double aik = get(A, n, i, k);
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += aik * get(B, n, k, j);
            }
        }
    }
}

// OpenMP 简单版本：对 i 循环并行
void matmul_omp_simple(const std::vector<double>& A,
                       const std::vector<double>& B,
                       std::vector<double>& C,
                       int n)
{
    std::fill(C.begin(), C.end(), 0.0);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double aik = get(A, n, i, k);
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += aik * get(B, n, k, j);
            }
        }
    }
}

// 带分块（blocking）的 OpenMP 并行矩阵乘
// blockSize: 分块大小，例如 32 或 64
void matmul_omp_blocked(const std::vector<double>& A,
                        const std::vector<double>& B,
                        std::vector<double>& C,
                        int n,
                        int blockSize)
{
    std::fill(C.begin(), C.end(), 0.0);

    // 防止 blockSize 异常
    if (blockSize <= 0 || blockSize > n) {
        blockSize = std::min(64, n);
    }

    // 在块级并行：ii, jj 两个维度 collapse，保证每个线程负责 C 的一块子矩阵
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += blockSize) {
        for (int jj = 0; jj < n; jj += blockSize) {

            for (int kk = 0; kk < n; kk += blockSize) {

                int iMax = std::min(ii + blockSize, n);
                int jMax = std::min(jj + blockSize, n);
                int kMax = std::min(kk + blockSize, n);

                for (int i = ii; i < iMax; ++i) {
                    for (int k = kk; k < kMax; ++k) {
                        double aik = get(A, n, i, k);
                        for (int j = jj; j < jMax; ++j) {
                            C[i * n + j] += aik * get(B, n, k, j);
                        }
                    }
                }

            } // kk
        } // jj
    } // ii
}

// 计算两个矩阵的最大差值，用于正确性验证
double max_diff(const std::vector<double>& X,
                const std::vector<double>& Y)
{
    double md = 0.0;
    if (X.size() != Y.size()) return 1e9;
    for (size_t i = 0; i < X.size(); ++i) {
        double d = std::fabs(X[i] - Y[i]);
        if (d > md) md = d;
    }
    return md;
}

int main(int argc, char** argv) {
    // ===================== 参数设置 =====================
    int n = 512;       // 默认矩阵维度
    int blockSize = 64; // 默认分块大小

    if (argc >= 2) {
        n = std::atoi(argv[1]);
        if (n <= 0) n = 512;
    }
    if (argc >= 3) {
        blockSize = std::atoi(argv[2]);
        if (blockSize <= 0) blockSize = 64;
    }

    std::cout << "=== OpenMP Matrix Multiplication Experiment ===\n";
    std::cout << "Matrix size: " << n << " x " << n << "\n";
    std::cout << "Block size (for blocked version): " << blockSize << "\n";

    int max_threads = omp_get_max_threads();
    std::cout << "Max available threads = " << max_threads << "\n";

    // ===================== 数据生成 =====================
    std::vector<double> A(n * n), B(n * n);
    std::vector<double> C_serial(n * n),
                        C_omp_simple(n * n),
                        C_omp_blocked(n * n);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < n * n; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    // ===================== 串行矩阵乘计时 =====================
    double t1 = omp_get_wtime();
    matmul_serial(A, B, C_serial, n);
    double t2 = omp_get_wtime();
    double t_serial = t2 - t1;
    std::cout << "[Serial] time = " << t_serial << " s\n";

    // ===================== OpenMP 简单并行计时 =====================
    double t3 = omp_get_wtime();
    matmul_omp_simple(A, B, C_omp_simple, n);
    double t4 = omp_get_wtime();
    double t_simple = t4 - t3;
    std::cout << "[OpenMP simple] time = " << t_simple << " s\n";

    double diff_simple = max_diff(C_serial, C_omp_simple);
    std::cout << "Max abs diff (serial vs simple) = " << diff_simple << "\n";

    // ===================== OpenMP 分块并行计时 =====================
    double t5 = omp_get_wtime();
    matmul_omp_blocked(A, B, C_omp_blocked, n, blockSize);
    double t6 = omp_get_wtime();
    double t_blocked = t6 - t5;
    std::cout << "[OpenMP blocked] time = " << t_blocked << " s\n";

    double diff_blocked = max_diff(C_serial, C_omp_blocked);
    std::cout << "Max abs diff (serial vs blocked) = " << diff_blocked << "\n";

    // ===================== 线程数、加速比与效率 =====================
    int threads_used = -1;
    #pragma omp parallel
    {
        #pragma omp master
        {
            threads_used = omp_get_num_threads();
        }
    }

    if (t_simple > 0.0) {
        double speedup_simple = t_serial / t_simple;
        double eff_simple = speedup_simple / threads_used;
        std::cout << "[OpenMP simple] Speedup S_p = " << speedup_simple
                  << ", Efficiency E_p = " << eff_simple << "\n";
    }

    if (t_blocked > 0.0) {
        double speedup_blocked = t_serial / t_blocked;
        double eff_blocked = speedup_blocked / threads_used;
        std::cout << "[OpenMP blocked] Speedup S_p = " << speedup_blocked
                  << ", Efficiency E_p = " << eff_blocked << "\n";
    }

    return 0;
}

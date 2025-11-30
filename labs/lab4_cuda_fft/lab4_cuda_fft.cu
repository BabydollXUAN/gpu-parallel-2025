#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA Error %s (%d): %s\n", #call, (int)err,     \
                    cudaGetErrorString(err));                                \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

#define CHECK_CUFFT(call)                                                    \
    do {                                                                     \
        cufftResult res = (call);                                            \
        if (res != CUFFT_SUCCESS) {                                          \
            fprintf(stderr, "cuFFT Error %s (%d)\n", #call, (int)res);       \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// 朴素 CPU DFT 实现：O(N^2)，只用于中等 N（比如 <= 16384）
void cpu_dft(const cufftComplex* in, cufftComplex* out, int N) {
    const float PI = 3.14159265358979323846f;
    for (int k = 0; k < N; ++k) {
        float real_sum = 0.0f;
        float imag_sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            float angle = -2.0f * PI * k * n / N;
            float c = std::cos(angle);
            float s = std::sin(angle);
            float xr = in[n].x;
            float xi = in[n].y;
            // (xr + j*xi) * (c + j*s)
            real_sum += xr * c - xi * s;
            imag_sum += xr * s + xi * c;
        }
        out[k].x = real_sum;
        out[k].y = imag_sum;
    }
}

int main(int argc, char** argv) {
    if (argc < 1 + 1) {
        std::fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int N = std::atoi(argv[1]);
    if (N <= 0) {
        std::fprintf(stderr, "Invalid N = %d\n", N);
        return EXIT_FAILURE;
    }

    std::printf("=== CUDA FFT Experiment ===\n");
    std::printf("N = %d\n", N);
    if (N > 20000) {
        std::printf("[Warning] N is large, CPU DFT (O(N^2)) may be slow. "
                    "Consider using N <= 16384 for experiments.\n");
    }

    // ----------------- 1. 分配主机内存并初始化输入 -----------------
    std::vector<cufftComplex> h_in(N);
    std::vector<cufftComplex> h_cpu_out(N);
    std::vector<cufftComplex> h_gpu_out(N);

    std::srand(42);
    for (int i = 0; i < N; ++i) {
        float r = static_cast<float>(std::rand()) / RAND_MAX; // 实部随机
        h_in[i].x = r;
        h_in[i].y = 0.0f;                                      // 虚部设为 0
    }

    // ----------------- 2. CPU 端 DFT 计时 -----------------
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_dft(h_in.data(), h_cpu_out.data(), N);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = t1 - t0;

    std::printf("[CPU DFT ] time = %.3f ms\n", cpu_ms.count());

    // ----------------- 3. GPU 端 cuFFT -----------------
    cufftComplex* d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, sizeof(cufftComplex) * N));

    // 拷贝输入到 GPU
    CHECK_CUDA(cudaMemcpy(d_data, h_in.data(),
                          sizeof(cufftComplex) * N,
                          cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 计时 cuFFT
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));

    // 将结果拷回主机
    CHECK_CUDA(cudaMemcpy(h_gpu_out.data(), d_data,
                          sizeof(cufftComplex) * N,
                          cudaMemcpyDeviceToHost));

    // ----------------- 4. 计算 Speedup 与误差 -----------------
    double speedup = cpu_ms.count() / gpu_ms;

    // 计算最大误差（欧氏范数）
    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float dr = h_cpu_out[i].x - h_gpu_out[i].x;
        float di = h_cpu_out[i].y - h_gpu_out[i].y;
        float err = std::sqrt(dr * dr + di * di);
        if (err > max_err) max_err = err;
    }

    std::printf("[GPU cuFFT] time = %.3f ms\n", gpu_ms);
    std::printf("Speedup S_p = %.3f\n", speedup);
    std::printf("Max abs diff between CPU DFT and GPU FFT = %.6f\n", max_err);

    // ----------------- 5. 清理资源 -----------------
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}

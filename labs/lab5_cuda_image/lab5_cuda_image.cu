#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA Error %s (%d): %s\n", #call, (int)err,     \
                    cudaGetErrorString(err));                                \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// -------------------- CPU 端 3x3 均值滤波 --------------------
void cpu_box_blur(const float* in, float* out, int width, int height)
{
    const float kernel[3][3] = {
        {1.f / 9, 1.f / 9, 1.f / 9},
        {1.f / 9, 1.f / 9, 1.f / 9},
        {1.f / 9, 1.f / 9, 1.f / 9}
    };

    // 简单处理：边界像素直接拷贝，不做卷积
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
                out[y * width + x] = in[y * width + x];
                continue;
            }
            float sum = 0.f;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    float w = kernel[ky + 1][kx + 1];
                    sum += w * in[(y + ky) * width + (x + kx)];
                }
            }
            out[y * width + x] = sum;
        }
    }
}

// -------------------- GPU naive 版本 --------------------
__global__ void box_blur_naive_kernel(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // 边界直接拷贝
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        out[y * width + x] = in[y * width + x];
        return;
    }

    const float kernel[3][3] = {
        {1.f / 9, 1.f / 9, 1.f / 9},
        {1.f / 9, 1.f / 9, 1.f / 9},
        {1.f / 9, 1.f / 9, 1.f / 9}
    };

    float sum = 0.f;
    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            float w = kernel[ky + 1][kx + 1];
            sum += w * in[(y + ky) * width + (x + kx)];
        }
    }
    out[y * width + x] = sum;
}

// -------------------- GPU shared memory 版本 --------------------
template<int BLOCK_SIZE>
__global__ void box_blur_shared_kernel(const float* __restrict__ in,
                                       float* __restrict__ out,
                                       int width, int height)
{
    // tile 包含 halo，因此多出 2
    __shared__ float tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;

    // 对应 tile 中心区域的坐标
    int tile_x = tx + 1;
    int tile_y = ty + 1;

    // Lambda：安全读取
    auto clamp = [](int v, int low, int high) {
        return v < low ? low : (v > high ? high : v);
    };

    // 先加载中心区域
    if (x < width && y < height) {
        int cx = clamp(x, 0, width - 1);
        int cy = clamp(y, 0, height - 1);
        tile[tile_y][tile_x] = in[cy * width + cx];
    }

    // 加载 halo：上下左右 + 四个角
    // 上边
    if (ty == 0) {
        int hy = clamp(y - 1, 0, height - 1);
        int hx = clamp(x,     0, width  - 1);
        tile[0][tile_x] = in[hy * width + hx];
    }
    // 下边
    if (ty == BLOCK_SIZE - 1) {
        int hy = clamp(y + 1, 0, height - 1);
        int hx = clamp(x,     0, width  - 1);
        tile[BLOCK_SIZE + 1][tile_x] = in[hy * width + hx];
    }
    // 左边
    if (tx == 0) {
        int hy = clamp(y,     0, height - 1);
        int hx = clamp(x - 1, 0, width  - 1);
        tile[tile_y][0] = in[hy * width + hx];
    }
    // 右边
    if (tx == BLOCK_SIZE - 1) {
        int hy = clamp(y,     0, height - 1);
        int hx = clamp(x + 1, 0, width  - 1);
        tile[tile_y][BLOCK_SIZE + 1] = in[hy * width + hx];
    }
    // 左上角
    if (tx == 0 && ty == 0) {
        int hy = clamp(y - 1, 0, height - 1);
        int hx = clamp(x - 1, 0, width  - 1);
        tile[0][0] = in[hy * width + hx];
    }
    // 右上角
    if (tx == BLOCK_SIZE - 1 && ty == 0) {
        int hy = clamp(y - 1, 0, height - 1);
        int hx = clamp(x + 1, 0, width  - 1);
        tile[0][BLOCK_SIZE + 1] = in[hy * width + hx];
    }
    // 左下角
    if (tx == 0 && ty == BLOCK_SIZE - 1) {
        int hy = clamp(y + 1, 0, height - 1);
        int hx = clamp(x - 1, 0, width  - 1);
        tile[BLOCK_SIZE + 1][0] = in[hy * width + hx];
    }
    // 右下角
    if (tx == BLOCK_SIZE - 1 && ty == BLOCK_SIZE - 1) {
        int hy = clamp(y + 1, 0, height - 1);
        int hx = clamp(x + 1, 0, width  - 1);
        tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = in[hy * width + hx];
    }

    __syncthreads();

    if (x >= width || y >= height) return;

    // 边界直接拷贝
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        out[y * width + x] = in[y * width + x];
        return;
    }

    const float kernel[3][3] = {
        {1.f / 9, 1.f / 9, 1.f / 9},
        {1.f / 9, 1.f / 9, 1.f / 9},
        {1.f / 9, 1.f / 9, 1.f / 9}
    };

    float sum = 0.f;
    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            float w = kernel[ky + 1][kx + 1];
            sum += w * tile[tile_y + ky][tile_x + kx];
        }
    }
    out[y * width + x] = sum;
}

// -------------------- 工具函数：计算最大绝对误差 --------------------
float max_abs_diff(const float* a, const float* b, int n)
{
    float mx = 0.f;
    for (int i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

// -------------------- main --------------------
int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <N> [BLOCK_SIZE]\n", argv[0]);
        return EXIT_FAILURE;
    }
    int N = std::atoi(argv[1]);
    int BLOCK_SIZE = (argc >= 3) ? std::atoi(argv[2]) : 16;

    if (N <= 0 || BLOCK_SIZE <= 0) {
        std::fprintf(stderr, "Invalid N or BLOCK_SIZE\n");
        return EXIT_FAILURE;
    }

    int width = N;
    int height = N;
    int num_pixels = width * height;

    std::printf("=== CUDA Image Blur Experiment ===\n");
    std::printf("Image size: %d x %d\n", width, height);
    std::printf("Block size: %d x %d\n", BLOCK_SIZE, BLOCK_SIZE);

    // 1. 分配主机内存并初始化“图像”
    std::vector<float> h_in(num_pixels);
    std::vector<float> h_cpu_out(num_pixels);
    std::vector<float> h_gpu_naive(num_pixels);
    std::vector<float> h_gpu_shared(num_pixels);

    // 简单生成一个梯度图像：从左上到右下逐渐变亮
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float val = (float(x) / (width - 1) + float(y) / (height - 1)) * 0.5f;
            h_in[y * width + x] = val;
        }
    }

    // 2. CPU 计时
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_box_blur(h_in.data(), h_cpu_out.data(), width, height);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dt_cpu = t1 - t0;
    double cpu_ms = dt_cpu.count();
    std::printf("[CPU ] time = %.3f ms\n", cpu_ms);

    // 3. 分配 GPU 内存
    float *d_in = nullptr, *d_out_naive = nullptr, *d_out_shared = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, sizeof(float) * num_pixels));
    CHECK_CUDA(cudaMalloc(&d_out_naive, sizeof(float) * num_pixels));
    CHECK_CUDA(cudaMalloc(&d_out_shared, sizeof(float) * num_pixels));

    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(),
                          sizeof(float) * num_pixels,
                          cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 4. GPU naive 版本计时
    cudaEvent_t start_naive, stop_naive;
    CHECK_CUDA(cudaEventCreate(&start_naive));
    CHECK_CUDA(cudaEventCreate(&stop_naive));

    CHECK_CUDA(cudaEventRecord(start_naive));
    box_blur_naive_kernel<<<grid, block>>>(d_in, d_out_naive, width, height);
    CHECK_CUDA(cudaEventRecord(stop_naive));
    CHECK_CUDA(cudaEventSynchronize(stop_naive));

    float naive_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&naive_ms, start_naive, stop_naive));

    CHECK_CUDA(cudaMemcpy(h_gpu_naive.data(), d_out_naive,
                          sizeof(float) * num_pixels,
                          cudaMemcpyDeviceToHost));

    float err_naive = max_abs_diff(h_cpu_out.data(), h_gpu_naive.data(), num_pixels);

    std::printf("[GPU naive ] time = %.3f ms, Speedup S_p = %.3f, max_err = %.6f\n",
                naive_ms, cpu_ms / naive_ms, err_naive);

    // 5. GPU shared 版本计时
    cudaEvent_t start_sh, stop_sh;
    CHECK_CUDA(cudaEventCreate(&start_sh));
    CHECK_CUDA(cudaEventCreate(&stop_sh));

    CHECK_CUDA(cudaEventRecord(start_sh));
    // 根据 BLOCK_SIZE 实例化模板
    if (BLOCK_SIZE == 8) {
        box_blur_shared_kernel<8><<<grid, block>>>(d_in, d_out_shared, width, height);
    } else if (BLOCK_SIZE == 16) {
        box_blur_shared_kernel<16><<<grid, block>>>(d_in, d_out_shared, width, height);
    } else if (BLOCK_SIZE == 32) {
        box_blur_shared_kernel<32><<<grid, block>>>(d_in, d_out_shared, width, height);
    } else {
        // 其它 block size 没有专门模板，先退回 naive 或报错
        printf("[Warning] BLOCK_SIZE=%d not supported for shared kernel, skip.\n", BLOCK_SIZE);
    }
    CHECK_CUDA(cudaEventRecord(stop_sh));
    CHECK_CUDA(cudaEventSynchronize(stop_sh));

    float shared_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&shared_ms, start_sh, stop_sh));

    CHECK_CUDA(cudaMemcpy(h_gpu_shared.data(), d_out_shared,
                          sizeof(float) * num_pixels,
                          cudaMemcpyDeviceToHost));

    float err_shared = max_abs_diff(h_cpu_out.data(), h_gpu_shared.data(), num_pixels);

    if (BLOCK_SIZE == 8 || BLOCK_SIZE == 16 || BLOCK_SIZE == 32) {
        std::printf("[GPU shared] time = %.3f ms, Speedup S_p = %.3f, max_err = %.6f\n",
                    shared_ms, cpu_ms / shared_ms, err_shared);
    }

    // 6. 清理资源
    CHECK_CUDA(cudaEventDestroy(start_naive));
    CHECK_CUDA(cudaEventDestroy(stop_naive));
    CHECK_CUDA(cudaEventDestroy(start_sh));
    CHECK_CUDA(cudaEventDestroy(stop_sh));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out_naive));
    CHECK_CUDA(cudaFree(d_out_shared));

    return 0;
}

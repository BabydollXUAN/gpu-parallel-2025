#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA Error %s (%d): %s\n",               \
                         #call, (int)err, cudaGetErrorString(err));        \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

// ----------------- PGM 图像读写辅助函数 -----------------
void save_pgm(const std::string &filename, const float *img, int W, int H) {
    FILE *f = std::fopen(filename.c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "P5\n%d %d\n255\n", W, H);
    for (int i = 0; i < W * H; ++i) {
        float v = img[i];
        v = std::max(0.0f, std::min(1.0f, v));
        unsigned char p = static_cast<unsigned char>(v * 255.0f + 0.5f);
        std::fwrite(&p, 1, 1, f);
    }
    std::fclose(f);
}

bool load_pgm(const std::string &filename, std::vector<float> &img, int &W, int &H) {
    FILE *f = std::fopen(filename.c_str(), "rb");
    if (!f) {
        std::perror("fopen");
        return false;
    }
    char magic[3] = {0};
    if (std::fscanf(f, "%2s", magic) != 1) return false;
    // 跳过注释
    int c = std::fgetc(f);
    while (c == '#') {
        while (c != '\n' && c != EOF) c = std::fgetc(f);
        c = std::fgetc(f);
    }
    std::ungetc(c, f);
    int maxval;
    if (std::fscanf(f, "%d %d %d", &W, &H, &maxval) != 3) return false;
    std::fgetc(f); // skip newline
    std::vector<unsigned char> buf(W * H);
    if (std::fread(buf.data(), 1, W * H, f) != (size_t)(W * H)) return false;
    std::fclose(f);
    img.resize(W * H);
    float scale = 1.0f / (float)maxval;
    for (int i = 0; i < W * H; ++i) img[i] = buf[i] * scale;
    return true;
}

// ----------------- 常量内存：存储 H逆矩阵 -----------------
__constant__ float d_Hinv[9];

// ★★★ 请确保这里的数值是你用 Python 算出来的真实参数 ★★★
__host__ void get_Hinv_host(float Hinv[9]) {
    // 示例数据，请用你 calibrated 出来的真实数据替换！
    // 如果你还没有更新，请去 Python 脚本里复制
    Hinv[0] = 0.25004793f; Hinv[1] = -0.63625292f; Hinv[2] = 265.65849615f;
    Hinv[3] = 0.02820959f; Hinv[4] = -0.39842684f; Hinv[5] = 210.47166917f;
    Hinv[6] = 0.00011805f; Hinv[7] = -0.00193882f; Hinv[8] = 1.00000000f;
}

// ----------------- 1. CPU 基线版本 -----------------
float bilinear_sample_cpu(const float *img, int W, int H, float x, float y) {
    if (x < 0.0f || y < 0.0f || x > W - 1.001f || y > H - 1.001f) return 0.0f;
    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float dx = x - x0;
    float dy = y - y0;
    // 简单的边界保护
    if (x1 >= W) x1 = W - 1;
    if (y1 >= H) y1 = H - 1;

    float v00 = img[y0 * W + x0];
    float v10 = img[y0 * W + x1];
    float v01 = img[y1 * W + x0];
    float v11 = img[y1 * W + x1];
    return (1 - dy) * ((1 - dx) * v00 + dx * v10) + dy * ((1 - dx) * v01 + dx * v11);
}

void ipm_cpu(const float *src, float *dst, int inW, int inH, int outW, int outH, const float Hinv[9]) {
    for (int v = 0; v < outH; ++v) {
        for (int u = 0; u < outW; ++u) {
            float x = Hinv[0] * u + Hinv[1] * v + Hinv[2];
            float y = Hinv[3] * u + Hinv[4] * v + Hinv[5];
            float w = Hinv[6] * u + Hinv[7] * v + Hinv[8];
            float src_x = x / w;
            float src_y = y / w;
            dst[v * outW + u] = bilinear_sample_cpu(src, inW, inH, src_x, src_y);
        }
    }
}

// ----------------- 2. GPU 版本 A: 全局内存 (Global Memory) -----------------
__device__ float bilinear_sample_gpu_manual(const float *img, int W, int H, float x, float y) {
    if (x < 0.0f || y < 0.0f || x > W - 1.001f || y > H - 1.001f) return 0.0f;
    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = min(x0 + 1, W - 1);
    int y1 = min(y0 + 1, H - 1);
    float dx = x - x0;
    float dy = y - y0;
    float v00 = img[y0 * W + x0];
    float v10 = img[y0 * W + x1];
    float v01 = img[y1 * W + x0];
    float v11 = img[y1 * W + x1];
    return (1 - dy) * ((1 - dx) * v00 + dx * v10) + dy * ((1 - dx) * v01 + dx * v11);
}

__global__ void ipm_kernel_global(const float *src, float *dst, int inW, int inH, int outW, int outH) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= outW || v >= outH) return;

    float x = d_Hinv[0] * u + d_Hinv[1] * v + d_Hinv[2];
    float y = d_Hinv[3] * u + d_Hinv[4] * v + d_Hinv[5];
    float w = d_Hinv[6] * u + d_Hinv[7] * v + d_Hinv[8];

    float src_x = x / w;
    float src_y = y / w;
    dst[v * outW + u] = bilinear_sample_gpu_manual(src, inW, inH, src_x, src_y);
}

// ----------------- 3. GPU 版本 B: 纹理内存 (Texture Memory) -----------------
// 优势：
// 1. 利用 Texture Cache，针对 2D 空间局部性优化，缓解非合并访存压力。
// 2. 利用硬件插值单元 (Texture Unit)，不占用 CUDA Core 计算资源。
// 3. 自动处理边界 (Border Mode)。
__global__ void ipm_kernel_texture(cudaTextureObject_t tex, float *dst, int outW, int outH) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= outW || v >= outH) return;

    // 坐标变换逻辑完全一致
    float x = d_Hinv[0] * u + d_Hinv[1] * v + d_Hinv[2];
    float y = d_Hinv[3] * u + d_Hinv[4] * v + d_Hinv[5];
    float w = d_Hinv[6] * u + d_Hinv[7] * v + d_Hinv[8];

    float src_x = x / w;
    float src_y = y / w;

    // 直接采样！
    // 注意：tex2D 使用非归一化坐标时，像素中心通常建议偏移 +0.5f 以获得最精确对齐
    // 越界部分会自动变为 0 (由 cudaAddressModeBorder 决定)
    dst[v * outW + u] = tex2D<float>(tex, src_x + 0.5f, src_y + 0.5f);
}

int main() {
    // 1. 读取输入
    int inW, inH;
    std::vector<float> h_input;
    if (!load_pgm("input.pgm", h_input, inW, inH)) {
        std::cerr << "Error: Cannot load input.pgm\n";
        return 1;
    }
    int outW = inW, outH = inH;

    std::cout << "=== IPM Optimization Experiment ===\n";
    std::cout << "Image Size: " << inW << " x " << inH << "\n";

    // 2. 准备输出缓冲区
    std::vector<float> h_ipm_cpu(outW * outH);
    std::vector<float> h_ipm_gpu_global(outW * outH);
    std::vector<float> h_ipm_gpu_tex(outW * outH);

    // 3. 拷贝矩阵到常量内存
    float Hinv_host[9];
    get_Hinv_host(Hinv_host);
    CHECK_CUDA(cudaMemcpyToSymbol(d_Hinv, Hinv_host, 9 * sizeof(float)));

    // ★★★ 修复点：在这里定义 dt，让它在整个 main 函数可见
    double dt = 0.0;

    // ==========================================
    // Part 1: CPU Baseline
    // ==========================================
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        ipm_cpu(h_input.data(), h_ipm_cpu.data(), inW, inH, outW, outH, Hinv_host);
        auto t1 = std::chrono::high_resolution_clock::now();
        // ★★★ 修复点：去掉前面的 double
        dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[CPU] Time: " << dt << " ms\n";
        save_pgm("ipm_result_cpu.pgm", h_ipm_cpu.data(), outW, outH);
    }

    // ==========================================
    // Part 2: GPU (Global Memory + Manual Interpolation)
    // ==========================================
    float *d_input, *d_output_global;
    CHECK_CUDA(cudaMalloc(&d_input, inW * inH * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_global, outW * outH * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), inW * inH * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    ipm_kernel_global<<<grid, block>>>(d_input, d_output_global, inW, inH, outW, outH);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_global = 0;
    cudaEventElapsedTime(&ms_global, start, stop);
    std::cout << "[GPU Global] Time: " << ms_global << " ms\n";
    
    CHECK_CUDA(cudaMemcpy(h_ipm_gpu_global.data(), d_output_global, outW * outH * sizeof(float), cudaMemcpyDeviceToHost));
    save_pgm("ipm_result_gpu_global.pgm", h_ipm_gpu_global.data(), outW, outH);
    
    // ==========================================
    // Part 3: GPU (Texture Memory + Hardware Interpolation)
    // ==========================================
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    CHECK_CUDA(cudaMallocArray(&cuArray, &channelDesc, inW, inH));
    CHECK_CUDA(cudaMemcpyToArray(cuArray, 0, 0, h_input.data(), inW * inH * sizeof(float), cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeBorder;
    texDesc.addressMode[1]   = cudaAddressModeBorder;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    float *d_output_tex = d_output_global; 

    cudaEventRecord(start);
    ipm_kernel_texture<<<grid, block>>>(texObj, d_output_tex, outW, outH);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_tex = 0;
    cudaEventElapsedTime(&ms_tex, start, stop);
    std::cout << "[GPU Texture] Time: " << ms_tex << " ms\n";

    CHECK_CUDA(cudaMemcpy(h_ipm_gpu_tex.data(), d_output_tex, outW * outH * sizeof(float), cudaMemcpyDeviceToHost));
    save_pgm("ipm_result_gpu_tex.pgm", h_ipm_gpu_tex.data(), outW, outH);

    // ==========================================
    // 结果分析
    // ==========================================
    std::cout << "------------------------------------------------\n";
    std::cout << "Speedup (CPU vs Global) : " << dt / ms_global << " x\n";
    std::cout << "Speedup (CPU vs Texture): " << dt / ms_tex << " x\n";
    std::cout << "Improvement (Global vs Texture): " << (ms_global - ms_tex) / ms_global * 100.0f << " %\n";

    CHECK_CUDA(cudaDestroyTextureObject(texObj));
    CHECK_CUDA(cudaFreeArray(cuArray));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output_global));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
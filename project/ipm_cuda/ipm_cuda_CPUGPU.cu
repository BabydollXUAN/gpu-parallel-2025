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

// ... 前面的 include, kernel, load_pgm 等保持不变 ...

// 辅助函数：在 CPU 端简单粗暴地放大图片 (Nearest Neighbor)
// 用于生成测试用的 1080P/4K 数据
std::vector<float> resize_image_cpu(const std::vector<float>& src, int w1, int h1, int w2, int h2) {
    std::vector<float> dst(w2 * h2);
    float x_ratio = (float)w1 / w2;
    float y_ratio = (float)h1 / h2;
    for (int i = 0; i < h2; i++) {
        for (int j = 0; j < w2; j++) {
            int px = (int)(j * x_ratio);
            int py = (int)(i * y_ratio);
            dst[i * w2 + j] = src[py * w1 + px];
        }
    }
    return dst;
}

int main() {
    // 1. 读取基础数据 (640x480)
    int baseW, baseH;
    std::vector<float> base_input;
    if (!load_pgm("input.pgm", base_input, baseW, baseH)) {
        std::cerr << "Error: Cannot load input.pgm\n";
        return 1;
    }

    // 2. 准备 H 矩阵
    float Hinv_host[9];
    get_Hinv_host(Hinv_host);
    CHECK_CUDA(cudaMemcpyToSymbol(d_Hinv, Hinv_host, 9 * sizeof(float)));

    // 定义测试分辨率
    struct TestConfig { std::string name; int w; int h; };
    std::vector<TestConfig> resolutions = {
        {"VGA  (640x480)",   640,  480},
        {"HD   (1280x720)",  1280, 720},
        {"FHD  (1920x1080)", 1920, 1080},
        {"4K   (3840x2160)", 3840, 2160}
    };

    std::cout << "===============================================================================\n";
    std::cout << "   IPM AUTOMATED BENCHMARK SUITE (CPU vs GPU Comparison)         \n";
    std::cout << "===============================================================================\n\n";

    // -------------------------------------------------------------
    // 实验 1：不同分辨率下的 CPU vs GPU 性能对比
    // -------------------------------------------------------------
    std::cout << ">>> EXPERIMENT 1: Resolution Impact & Speedup Analysis\n";
    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << "| Resolution      |   Pixels  | CPU Time (ms) | GPU Time (ms) | Speedup (x) |\n";
    std::cout << "|-----------------|-----------|---------------|---------------|-------------|\n";

    // 用于保存 1080P 下的 CPU 时间，供实验 2 参考
    double fhd_cpu_time = 0.0; 

    for (const auto& res : resolutions) {
        // A. 准备数据 (自动放大)
        std::vector<float> h_input = resize_image_cpu(base_input, baseW, baseH, res.w, res.h);
        
        // --- [新增] CPU 跑分部分 ---
        // 注意：4K 分辨率下 CPU 可能会跑几百毫秒，请耐心等待
        std::vector<float> h_output_cpu(res.w * res.h); // 临时输出buffer
        auto t0_cpu = std::chrono::high_resolution_clock::now();
        ipm_cpu(h_input.data(), h_output_cpu.data(), res.w, res.h, res.w, res.h, Hinv_host);
        auto t1_cpu = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1_cpu - t0_cpu).count();

        if (res.w == 1920) fhd_cpu_time = cpu_ms; // 记录一下 FHD 的时间

        // --- GPU 准备部分 ---
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaArray* cuArray;
        CHECK_CUDA(cudaMallocArray(&cuArray, &channelDesc, res.w, res.h));
        
        // 使用 cudaMemcpy2DToArray
        CHECK_CUDA(cudaMemcpy2DToArray(cuArray, 0, 0, 
                                       h_input.data(), res.w * sizeof(float), 
                                       res.w * sizeof(float), res.h, 
                                       cudaMemcpyHostToDevice));

        struct cudaResourceDesc resDesc; memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray; resDesc.res.array.array = cuArray;
        struct cudaTextureDesc texDesc; memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap; texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode     = cudaFilterModeLinear; texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaTextureObject_t texObj = 0;
        CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

        float *d_output;
        CHECK_CUDA(cudaMalloc(&d_output, res.w * res.h * sizeof(float)));

        // --- GPU 跑分部分 ---
        dim3 block(16, 16);
        dim3 grid((res.w + block.x - 1) / block.x, (res.h + block.y - 1) / block.y);

        // 预热
        ipm_kernel_texture<<<grid, block>>>(texObj, d_output, res.w, res.h);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        // GPU 跑 10 次取平均
        int n_iter = 10;
        for(int i=0; i<n_iter; ++i) {
            ipm_kernel_texture<<<grid, block>>>(texObj, d_output, res.w, res.h);
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float total_ms = 0;
        cudaEventElapsedTime(&total_ms, start, stop);
        float gpu_ms = total_ms / n_iter;
        
        // 计算加速比
        double speedup = cpu_ms / gpu_ms;

        // 打印结果
        printf("| %-15s | %9d | %13.4f | %13.4f | %11.1f |\n", 
               res.name.c_str(), res.w * res.h, cpu_ms, gpu_ms, speedup);

        // 清理
        CHECK_CUDA(cudaDestroyTextureObject(texObj));
        CHECK_CUDA(cudaFreeArray(cuArray));
        CHECK_CUDA(cudaFree(d_output));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }
    std::cout << "-------------------------------------------------------------------------------\n\n";

    // -------------------------------------------------------------
    // 实验 2：Block Size 影响分析 (固定在 FHD 1080P 下进行)
    // -------------------------------------------------------------
    std::cout << ">>> EXPERIMENT 2: Block Size Analysis (at 1920x1080)\n";
    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << "| Block Config | Threads | GPU Time (ms) | Speedup (vs CPU) | Occupancy Hint |\n";
    std::cout << "|--------------|---------|---------------|------------------|----------------|\n";

    int w = 1920, h = 1080;
    std::vector<float> h_input = resize_image_cpu(base_input, baseW, baseH, w, h);
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    CHECK_CUDA(cudaMallocArray(&cuArray, &channelDesc, w, h));
    CHECK_CUDA(cudaMemcpy2DToArray(cuArray, 0, 0, h_input.data(), w*sizeof(float), w*sizeof(float), h, cudaMemcpyHostToDevice));
    
    struct cudaResourceDesc resDesc; memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray; resDesc.res.array.array = cuArray;
    struct cudaTextureDesc texDesc; memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap; texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear; texDesc.readMode = cudaReadModeElementType; texDesc.normalizedCoords = 0;
    cudaTextureObject_t texObj; CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    float *d_output; CHECK_CUDA(cudaMalloc(&d_output, w * h * sizeof(float)));

    struct BlockConfig { int x; int y; std::string desc; };
    std::vector<BlockConfig> blocks = {
        {16, 16, "Standard (256)"},
        {32,  8, "Wide     (256)"},
        {32, 32, "Max      (1024)"},
        { 8,  8, "Small    (64) "}
    };

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    for (const auto& b : blocks) {
        dim3 block(b.x, b.y);
        dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

        cudaEventRecord(start);
        for(int i=0; i<10; ++i) {
            ipm_kernel_texture<<<grid, block>>>(texObj, d_output, w, h);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float total_ms = 0;
        cudaEventElapsedTime(&total_ms, start, stop);
        float gpu_ms = total_ms / 10.0f;
        
        // 这里用之前存下来的 fhd_cpu_time 来算相对加速比
        double speedup = fhd_cpu_time / gpu_ms;

        printf("| %2d x %-2d      | %7d | %13.4f | %16.1f | %-14s |\n", 
               b.x, b.y, b.x*b.y, gpu_ms, speedup, b.desc.c_str());
    }

    CHECK_CUDA(cudaDestroyTextureObject(texObj));
    CHECK_CUDA(cudaFreeArray(cuArray));
    CHECK_CUDA(cudaFree(d_output));
    
    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << "Benchmark Finished.\n";

    return 0;
}


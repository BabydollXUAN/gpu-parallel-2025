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

// ----------------- 简单 PGM 存图 -----------------
void save_pgm(const std::string &filename,
              const float *img, int W, int H) {
    FILE *f = std::fopen(filename.c_str(), "wb");
    if (!f) {
        std::perror("fopen");
        return;
    }
    // P5 = binary gray
    std::fprintf(f, "P5\n%d %d\n255\n", W, H);
    for (int i = 0; i < W * H; ++i) {
        float v = img[i];
        v = std::max(0.0f, std::min(1.0f, v));
        unsigned char p = static_cast<unsigned char>(v * 255.0f + 0.5f);
        std::fwrite(&p, 1, 1, f);
    }
    std::fclose(f);
}
// ----------------- 读取 PGM 到 float [0,1] -----------------
bool load_pgm(const std::string &filename,
              std::vector<float> &img,
              int &W, int &H) {
    FILE *f = std::fopen(filename.c_str(), "rb");
    if (!f) {
        std::perror("fopen");
        return false;
    }

    char magic[3] = {0};
    if (std::fscanf(f, "%2s", magic) != 1) {
        std::fprintf(stderr, "Failed to read magic from %s\n", filename.c_str());
        std::fclose(f);
        return false;
    }
    if (std::string(magic) != "P5") {
        std::fprintf(stderr, "%s is not P5 PGM (magic=%s)\n",
                     filename.c_str(), magic);
        std::fclose(f);
        return false;
    }

    // 跳过注释行
    int c = std::fgetc(f);
    while (c == '#') {
        while (c != '\n' && c != EOF) {
            c = std::fgetc(f);
        }
        c = std::fgetc(f);
    }
    std::ungetc(c, f);

    int maxval = 255;
    if (std::fscanf(f, "%d %d", &W, &H) != 2) {
        std::fprintf(stderr, "Failed to read width/height from %s\n", filename.c_str());
        std::fclose(f);
        return false;
    }
    if (std::fscanf(f, "%d", &maxval) != 1) {
        std::fprintf(stderr, "Failed to read maxval from %s\n", filename.c_str());
        std::fclose(f);
        return false;
    }

    // 吃掉一个换行
    std::fgetc(f);

    std::vector<unsigned char> buf(W * H);
    size_t n = std::fread(buf.data(), 1, W * H, f);
    std::fclose(f);
    if (n != (size_t)(W * H)) {
        std::fprintf(stderr, "PGM data size mismatch in %s\n", filename.c_str());
        return false;
    }

    img.resize(W * H);
    float scale = 1.0f / (float)maxval;
    for (int i = 0; i < W * H; ++i) {
        img[i] = buf[i] * scale;
    }
    return true;
}



// ----------------- 生成合成道路场景 -----------------
// 分辨率 640x480：底部宽、上部窄的车道，带两条亮车道线
void generate_synthetic_road(float *img, int W, int H) {
    std::fill(img, img + W * H, 0.0f);

    // 定义“远处车道上边缘”和“近处车道下边缘”的四个顶点
    // 这些点跟我们预先算好的 H 对应着（大概是个梯形车道）
    float top_y    = 300.0f;
    float bottom_y = 480.0f;
    float top_left_x   = 260.0f;
    float top_right_x  = 380.0f;
    float bottom_left_x  = 200.0f;
    float bottom_right_x = 440.0f;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float val = 0.0f;

            if (y >= top_y && y <= bottom_y) {
                float t = (y - top_y) / (bottom_y - top_y + 1e-6f);
                float xL = top_left_x  + (bottom_left_x  - top_left_x)  * t;
                float xR = top_right_x + (bottom_right_x - top_right_x) * t;

                // 车道区域略微亮一点
                if (x >= xL && x <= xR) {
                    val = 0.2f;
                }
                // 左右车道线更亮
                if (std::fabs(x - xL) < 2.0f || std::fabs(x - xR) < 2.0f) {
                    val = 1.0f;
                }
            }

            img[y * W + x] = val;
        }
    }
}

// ----------------- 预计算好的逆透视矩阵 H^{-1} -----------------
// 这是 3x3 矩阵，把“相机视角图像”上的车道梯形映射到“鸟瞰矩形”
// H_inv * [u; v; 1] = [x; y; w]，再用 (x/w, y/w) 去原图采样
//
// 这些系数是我提前用 4 对对应点算好的，假设输入大小 640x480、输出同尺寸。
__constant__ float d_Hinv[9];

// __host__ void get_Hinv_host(float Hinv[9]) {
//     // row-major: h00, h01, h02, h10, h11, h12, h20, h21, h22
//     Hinv[0] = -0.328125f;  Hinv[1] =  0.2916667f; Hinv[2] = -175.0f;
//     Hinv[3] =  0.0f;       Hinv[4] =  0.1276042f; Hinv[5] = -262.5f;
//     Hinv[6] =  0.0f;       Hinv[7] =  0.00091146f;Hinv[8] = -0.875f;
// }

__host__ void get_Hinv_host(float Hinv[9]) {
    // row-major: h00, h01, h02, h10, h11, h12, h20, h21, h22
    Hinv[0] = -0.6538462f;   Hinv[1] =  0.7005495f;   Hinv[2] = -336.2637363f;
    Hinv[3] =  0.0f;         Hinv[4] =  0.3969780f;   Hinv[5] = -471.7032967f;
    Hinv[6] =  0.0f;         Hinv[7] =  0.0023352f;   Hinv[8] = -1.7747253f;
}


// ----------------- CPU 双线性插值采样 -----------------
float bilinear_sample_cpu(const float *img, int W, int H,
                          float x, float y) {
    if (x < 0.0f || y < 0.0f || x > W - 1.0f || y > H - 1.0f) {
        return 0.0f;
    }
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = std::min(x0 + 1, W - 1);
    int y1 = std::min(y0 + 1, H - 1);
    float dx = x - x0;
    float dy = y - y0;

    float v00 = img[y0 * W + x0];
    float v10 = img[y0 * W + x1];
    float v01 = img[y1 * W + x0];
    float v11 = img[y1 * W + x1];

    float v0 = v00 + dx * (v10 - v00);
    float v1 = v01 + dx * (v11 - v01);
    return v0 + dy * (v1 - v0);
}

// ----------------- CPU 逆透视 -----------------
void ipm_cpu(const float *src, float *dst,
             int inW, int inH, int outW, int outH,
             const float Hinv[9]) {
    for (int v = 0; v < outH; ++v) {
        for (int u = 0; u < outW; ++u) {
            float U = static_cast<float>(u);
            float V = static_cast<float>(v);

            float x = Hinv[0] * U + Hinv[1] * V + Hinv[2];
            float y = Hinv[3] * U + Hinv[4] * V + Hinv[5];
            float w = Hinv[6] * U + Hinv[7] * V + Hinv[8];

            float src_x = x / w;
            float src_y = y / w;

            dst[v * outW + u] = bilinear_sample_cpu(src, inW, inH, src_x, src_y);
        }
    }
}

// ----------------- GPU 版本：双线性插值 -----------------
__device__ float bilinear_sample_gpu(const float *img,
                                     int W, int H,
                                     float x, float y) {
    if (x < 0.0f || y < 0.0f || x > W - 1.0f || y > H - 1.0f) {
        return 0.0f;
    }
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = min(x0 + 1, W - 1);
    int y1 = min(y0 + 1, H - 1);
    float dx = x - x0;
    float dy = y - y0;

    float v00 = img[y0 * W + x0];
    float v10 = img[y0 * W + x1];
    float v01 = img[y1 * W + x0];
    float v11 = img[y1 * W + x1];

    float v0 = v00 + dx * (v10 - v00);
    float v1 = v01 + dx * (v11 - v01);
    return v0 + dy * (v1 - v0);
}

// ----------------- GPU kernel：每个线程负责一个输出像素 -----------------
__global__ void ipm_kernel(const float *src, float *dst,
                           int inW, int inH, int outW, int outH) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= outW || v >= outH) return;

    float U = (float)u;
    float V = (float)v;

    float x = d_Hinv[0] * U + d_Hinv[1] * V + d_Hinv[2];
    float y = d_Hinv[3] * U + d_Hinv[4] * V + d_Hinv[5];
    float w = d_Hinv[6] * U + d_Hinv[7] * V + d_Hinv[8];

    float src_x = x / w;
    float src_y = y / w;

    float val = bilinear_sample_gpu(src, inW, inH, src_x, src_y);
    dst[v * outW + u] = val;
}

int main() {
    // const int inW  = 640;
    // const int inH  = 480;
    // const int outW = 640;
    // const int outH = 480;

    // std::cout << "=== CUDA IPM (Inverse Perspective Mapping) Experiment ===\n";
    // std::cout << "Input size  : " << inW  << " x " << inH  << "\n";
    // std::cout << "Output size : " << outW << " x " << outH << "\n";

    // // 1. Host 内存
    // std::vector<float> h_input(inW * inH);
    // std::vector<float> h_ipm_cpu(outW * outH);
    // std::vector<float> h_ipm_gpu(outW * outH);

    // // 2. 生成合成道路图
    // generate_synthetic_road(h_input.data(), inW, inH);
    // save_pgm("input.pgm", h_input.data(), inW, inH);
    int inW = 0, inH = 0;
    int outW = 0, outH = 0;

    // 1. 从 input.pgm 读入灰度图
    std::vector<float> h_input;
    if (!load_pgm("input.pgm", h_input, inW, inH)) {
        std::cerr << "Failed to load input.pgm, 请先用 Python 生成它\n";
        return 1;
    }
    outW = inW;
    outH = inH;

    std::cout << "=== CUDA IPM (Inverse Perspective Mapping) Experiment ===\n";
    std::cout << "Input size  : " << inW  << " x " << inH  << "\n";
    std::cout << "Output size : " << outW << " x " << outH << "\n";

    std::vector<float> h_ipm_cpu(outW * outH);
    std::vector<float> h_ipm_gpu(outW * outH);

    // （注意：这里不要再调用 generate_synthetic_road 了）

    // 3. 准备 H^{-1}，拷贝到 constant memory
    float Hinv_host[9];
    get_Hinv_host(Hinv_host);
    CHECK_CUDA(cudaMemcpyToSymbol(d_Hinv, Hinv_host, 9 * sizeof(float)));

    // 4. CPU 计算 + 计时
    auto t0 = std::chrono::high_resolution_clock::now();
    ipm_cpu(h_input.data(), h_ipm_cpu.data(),
            inW, inH, outW, outH, Hinv_host);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 5. GPU 计算
    float *d_input = nullptr;
    float *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input,  inW * inH  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, outW * outH * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(),
                          inW * inH * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x,
              (outH + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    ipm_kernel<<<grid, block>>>(d_input, d_output,
                                inW, inH, outW, outH);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_ipm_gpu.data(), d_output,
                          outW * outH * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // 6. 误差与加速比
    float max_err = 0.0f;
    for (int i = 0; i < outW * outH; ++i) {
        float e = std::fabs(h_ipm_cpu[i] - h_ipm_gpu[i]);
        if (e > max_err) max_err = e;
    }

    double speedup = cpu_ms / gpu_ms;

    std::cout << "[CPU ] IPM time = " << cpu_ms << " ms\n";
    std::cout << "[GPU ] IPM time = " << gpu_ms << " ms\n";
    std::cout << "Speedup S_p    = " << speedup << "\n";
    std::cout << "Max abs diff   = " << max_err << "\n";

    // 7. 存结果图像
    save_pgm("ipm_cpu.pgm", h_ipm_cpu.data(), outW, outH);
    save_pgm("ipm_gpu.pgm", h_ipm_gpu.data(), outW, outH);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    std::cout << "Saved input.pgm, ipm_cpu.pgm, ipm_gpu.pgm\n";
    return 0;
}

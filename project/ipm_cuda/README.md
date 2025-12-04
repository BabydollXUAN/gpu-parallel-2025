
# CUDA-Accelerated Inverse Perspective Mapping (IPM)

> **GPU Parallel Computing Course Final Project (2025)**

  

## 📖 Introduction (项目简介)

此项目实现了一个高性能的\*\*逆透视映射（Inverse Perspective Mapping, IPM）\*\*系统，广泛应用于自动驾驶领域的车道线检测与路面分析。

通过利用 NVIDIA CUDA 并行计算架构，本项目将原本在 CPU 上运行缓慢的透视变换算法移植到了 GPU 上。不仅实现了基础的 Global Memory 版本，还进一步利用 **Texture Memory（纹理内存）** 和 **硬件插值单元** 进行了深度优化。

项目包含一个全自动的 **Benchmark Suite**，能够测试从 VGA 到 4K 分辨率下的性能表现，并分析不同线程块（Block Size）配置对 GPU 占用率的影响。

## ✨ Key Features (核心特性)

  * **Hybrid Implementation**: 包含 CPU（串行基准）与 GPU（并行加速）的双版本实现，方便进行精确度与性能对比。
  * **Texture Memory Optimization**: 利用 GPU 纹理缓存（L1/Texture Cache）优化非合并访存（Non-coalesced access），并使用硬件双线性插值替代软件计算，显著提升吞吐量。
  * **Real-world Calibration**: 提供 Python 辅助脚本，支持针对真实路面图片的单应性矩阵（Homography Matrix）交互式标定。
  * **Automated Benchmark**: 内置自动化测试套件，实时生成分辨率压力测试报告与线程配置分析报告。
  * **High Performance**: 在 4K 分辨率下实现超过 **360 GPixels/s** 的吞吐量，相比 CPU 基线实现 **200+ 倍** 加速。

## 📂 Project Structure (目录结构)

```text
.
├── ipm_cuda.cu          # 核心源码 (包含 CPU, GPU Global, GPU Texture 及 Benchmark)
├── calib_h.py           # Python 标定脚本 (用于生成 H 逆矩阵)
├── convert_img.py       # 辅助脚本 (JPG/PNG -> PGM 格式转换)
├── input.pgm            # 输入测试图像 (需为 P5 PGM 格式)
├── README.md            # 项目文档
└── bin/                 # 编译输出目录
    └── ipm_cuda         # 可执行文件
```

## 🚀 Quick Start (快速开始)

### 1\. 环境依赖 (Prerequisites)

  * **Hardware**: NVIDIA GPU (Compute Capability \>= 3.5)
  * **Software**:
      * CUDA Toolkit (nvcc)
      * Python 3.x (NumPy, OpenCV)

### 2\. 图像标定 (Calibration)

在本地运行 Python 脚本，对真实路面图片进行四个角点的标定，获取变换矩阵参数。

```bash
# 1. 转换图片格式 (如果原图是 jpg/png)
python convert_img.py my_road.jpg

# 2. 运行标定工具
python calib_h.py input.pgm
```

*操作提示：依次点击图像中的 [左上 -\> 右上 -\> 右下 -\> 左下] 四个点。脚本将输出一段 C++ 代码，请将其复制并替换 `ipm_cuda.cu` 中的 `get_Hinv_host` 函数内容。*

### 3\. 编译 (Compilation)

在服务器端使用 `nvcc` 进行编译。开启 `-O2` 优化以获得最佳性能。

```bash
nvcc -O2 -std=c++14 -o ipm_cuda ipm_cuda.cu
```

### 4\. 运行 (Execution)

执行编译后的程序。程序将自动运行基准测试并输出性能报告。

```bash
./ipm_cuda
```

## 📊 Performance Analysis (性能分析)

以下数据基于 NVIDIA GPU (AutoDL环境) 实测结果。

### 1\. 分辨率对吞吐量的影响 (Resolution vs Throughput)

利用纹理内存优化后，随着分辨率提升，GPU 的计算吞吐量（MP/s）呈现上升趋势，证明算法在大规模数据下具有良好的可扩展性。

| Resolution | Pixels | GPU Time (ms) | Speedup (vs CPU) | Throughput (MP/s) |
| :--- | :--- | :--- | :--- | :--- |
| **VGA (640x480)** | 0.3 M | 0.0034 | \~130x | 90,909 |
| **HD (720P)** | 0.9 M | 0.0048 | \~150x | 191,489 |
| **FHD (1080P)** | 2.0 M | 0.0078 | \~190x | 266,447 |
| **4K (2160P)** | 8.3 M | **0.0226** | **\>200x** | **366,515** |

### 2\. 优化效果对比 (Optimization Impact)

对比 GPU 全局内存版本（Manual Interpolation）与纹理内存版本（Hardware Interpolation）：

  * **Global Memory**: 存在大量非合并内存读取，Cache 命中率低。
  * **Texture Memory**: 利用 2D 空间局部性缓存，且无需手动计算插值权重。
  * **结果**: 纹理内存版本在 1080P 分辨率下比全局内存版本快约 **40%**。

### 3\. 线程块配置分析 (Block Configuration)

在 1920x1080 分辨率下的测试结果表明，**32x8** 或 **16x16** (256 threads) 是最佳配置。

| Block Config | Time (ms) | Analysis |
| :--- | :--- | :--- |
| **32 x 8** | **0.0077** | **Best Performance**. 完美匹配 Warp 宽度，利于合并写入。 |
| **16 x 16** | 0.0088 | Good. 标准配置，性能略低于 32x8。 |
| **32 x 32** | 0.0109 | Slower. 1024 线程导致寄存器压力过大，降低了 Occupancy。 |
| **8 x 8** | 0.0184 | Worst. 线程过少，无法有效掩盖访存延迟。 |

## 🖼️ Results (效果展示)

| Input (Front View) | Output (Bird's Eye View) |
| :---: | :---: |
| *(Original Image)* | *(Transformed Image)* |
|  |  |

*(注：输出图像展示了经过纹理内存加速处理后的真实路面鸟瞰图，车道线已成功校正为平行状态。)*

## 📝 License

此项目仅供课程学习与交流使用。


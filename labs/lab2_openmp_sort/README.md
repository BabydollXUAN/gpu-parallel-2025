# Lab2: OpenMP Merge Sort

- `lab2_openmp_sort.cpp`: serial + OpenMP task-based merge sort
- `run.sh`: compile and run with `g++ -O2 -fopenmp`
- `results_N1e6_threshold20000.csv`: experimental results for N=1e6
  with different numbers of threads (speedup and efficiency).

## Lab2：基于 OpenMP 的排序并行与加速比分析

本实验围绕“排序算法的并行及优化”展开，选择 **归并排序（Merge Sort）** 作为核心算法，对比实现了：

- 串行归并排序 `mergesort_serial`
- 基于 **OpenMP task** 的并行归并排序 `mergesort_parallel`

并通过不同数据规模和线程数下的实验，分析了加速比与并行效率。

### 实验目标

- 利用 OpenMP 将分治结构的归并排序进行任务级并行化；
- 通过调整线程数和任务粒度（`threshold`），观察并行程序相对于串行程序的加速比变化；
- 理解并行开销、串行部分、内存带宽等因素对实际加速效果的影响。

### 算法与并行设计

- 串行版本使用经典递归归并排序，对区间 `[l, r)` 进行分治和合并。
- 并行版本在递归过程中对左右子区间创建 OpenMP `task`，通过：
  - 任务粒度阈值 `threshold`
  - 最大递归深度 `max_depth`
  
  控制 task 数量，避免过度拆分带来的调度开销。区间过小或递归过深时退化为串行排序。

### 实验设置

- 随机生成整数数组，分别测试数据规模：
  - `N = 1e6`, `N = 5e6`, `N = 1e7`
- 线程数设置：
  - `OMP_NUM_THREADS = 1, 2, 4, 8`
- 使用 `omp_get_wtime()` 分别测量：
  - 串行时间 `T_serial`
  - 并行时间 `T_parallel`
  - 加速比 `S_p = T_serial / T_parallel`
  - 并行效率 `E_p = S_p / p`

实验结果存放于：

- `results_N1e6_threshold20000.csv`
- `results_N5e6_threshold20000.csv`
- `results_N1e7_threshold50000.csv`

并通过 `plot_speedup.py` 生成 **Speedup vs Threads** 曲线图（保存在 `figures/` 目录）。

### 使用方法

在本目录下：

```bash
# 编译并运行（默认 N=1e6）
bash run.sh

# 指定数据规模和任务阈值，例如 N=1e7, threshold=50000
export OMP_NUM_THREADS=8
bash run.sh 10000000 50000

# 根据某个 CSV 画 Speedup 曲线
python plot_speedup.py results_N1e6_threshold20000.csv --output figures/speedup_N1e6.png

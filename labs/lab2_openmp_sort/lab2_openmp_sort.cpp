#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <omp.h>

// 串行 merge 操作：合并 [l, mid) 和 [mid, r)
void merge_range(std::vector<int>& a, int l, int mid, int r, std::vector<int>& tmp) {
    int i = l, j = mid, k = l;
    while (i < mid && j < r) {
        if (a[i] <= a[j]) tmp[k++] = a[i++];
        else              tmp[k++] = a[j++];
    }
    while (i < mid) tmp[k++] = a[i++];
    while (j < r)   tmp[k++] = a[j++];
    for (int t = l; t < r; ++t) {
        a[t] = tmp[t];
    }
}

// 串行归并排序
void mergesort_serial(std::vector<int>& a, int l, int r, std::vector<int>& tmp) {
    if (r - l <= 1) return;
    int mid = (l + r) / 2;
    mergesort_serial(a, l, mid, tmp);
    mergesort_serial(a, mid, r, tmp);
    merge_range(a, l, mid, r, tmp);
}

// 并行归并排序（使用 OpenMP task）
// depth 用来控制递归深度，threshold 控制任务粒度
void mergesort_parallel(std::vector<int>& a, int l, int r,
                        std::vector<int>& tmp, int depth,
                        int threshold, int max_depth) {
    if (r - l <= 1) return;

    // 区间太小或递归太深，就退化成串行，避免 task 过多
    if ((r - l) < threshold || depth >= max_depth) {
        mergesort_serial(a, l, r, tmp);
        return;
    }

    int mid = (l + r) / 2;

    #pragma omp task shared(a, tmp)
    {
        mergesort_parallel(a, l, mid, tmp, depth + 1, threshold, max_depth);
    }

    #pragma omp task shared(a, tmp)
    {
        mergesort_parallel(a, mid, r, tmp, depth + 1, threshold, max_depth);
    }

    #pragma omp taskwait
    merge_range(a, l, mid, r, tmp);
}

int main(int argc, char** argv) {
    // ===================== 参数设置 =====================
    int N = 1'000'000;           // 默认 1e6 个元素
    if (argc >= 2) {
        N = std::atoi(argv[1]);
        if (N <= 0) N = 1'000'000;
    }

    int threshold = 20'000;      // task 粒度阈值，区间长度小于该值就串行
    if (argc >= 3) {
        int t = std::atoi(argv[2]);
        if (t > 0) threshold = t;
    }

    int max_depth = 20;          // 最大 task 递归深度，防止过度拆分

    std::cout << "=== OpenMP sort experiment ===" << std::endl;
    std::cout << "N = " << N << ", threshold = " << threshold << std::endl;

    int max_threads = omp_get_max_threads();
    std::cout << "Max available threads (omp_get_max_threads) = "
              << max_threads << std::endl;

    // ===================== 数据生成 =====================
    std::vector<int> base(N);
    std::mt19937 rng(42); // 固定种子，方便复现实验
    std::uniform_int_distribution<int> dist(0, 1'000'000);

    for (int i = 0; i < N; ++i) {
        base[i] = dist(rng);
    }

    // 为串行和并行准备两份拷贝
    std::vector<int> arr_serial = base;
    std::vector<int> arr_parallel = base;

    std::vector<int> tmp_serial(N);
    std::vector<int> tmp_parallel(N);

    // ===================== 串行排序计时 =====================
    double t1 = omp_get_wtime();
    mergesort_serial(arr_serial, 0, N, tmp_serial);
    double t2 = omp_get_wtime();
    double serial_time = t2 - t1;

    std::cout << "[Serial] time = " << serial_time << " s" << std::endl;

    // ===================== 并行排序计时 =====================
    double t3 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            mergesort_parallel(arr_parallel, 0, N, tmp_parallel,
                               /*depth=*/0, threshold, max_depth);
        }
    }
    double t4 = omp_get_wtime();
    double parallel_time = t4 - t3;

    std::cout << "[Parallel] time = " << parallel_time << " s" << std::endl;

    // ===================== 正确性检查 =====================
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (arr_serial[i] != arr_parallel[i]) {
            ok = false;
            std::cerr << "Mismatch at index " << i << ": serial = "
                      << arr_serial[i] << ", parallel = " << arr_parallel[i] << std::endl;
            break;
        }
    }
    std::cout << "Correctness check: " << (ok ? "PASS" : "FAIL") << std::endl;

    // ===================== 加速比与效率 =====================
    if (parallel_time > 0) {
        double speedup = serial_time / parallel_time;
        int threads_used = -1;

        #pragma omp parallel
        {
            #pragma omp master
            {
                threads_used = omp_get_num_threads();
            }
        }

        std::cout << "Speedup S_p = " << speedup << std::endl;
        if (threads_used > 0) {
            double efficiency = speedup / threads_used;
            std::cout << "Threads used = " << threads_used
                      << ", Efficiency E_p = " << efficiency << std::endl;
        }
    }

    return 0;
}

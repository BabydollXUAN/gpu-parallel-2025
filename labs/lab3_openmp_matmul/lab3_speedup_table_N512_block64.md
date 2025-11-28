# Lab3 Speedup Table (N = 512, blockSize = 64)

| Threads (p) | Speedup_simple | Speedup_blocked |
|------------:|---------------:|----------------:|
| 1           | 1.11           | 1.36            |
| 2           | 1.50           | 2.72            |
| 4           | 2.52           | 5.40            |
| 8           | 3.58           | 8.56            |

> Notes:
> - Data from results_matmul_N512_block64.csv.
> - Matrix size: 512 × 512, blockSize = 64.
> - All runs use the same environment as described in Lab1.

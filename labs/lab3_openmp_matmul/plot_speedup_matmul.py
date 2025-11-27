import argparse
import csv

import matplotlib.pyplot as plt


def load_speedup_from_csv(path):
    threads = []
    speedup_simple = []
    speedup_blocked = []
    N = None
    block_size = None

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = int(row["threads"])
            s_simple = float(row["Speedup_simple"])
            s_blocked = float(row["Speedup_blocked"])

            threads.append(t)
            speedup_simple.append(s_simple)
            speedup_blocked.append(s_blocked)

            if N is None:
                N = int(row["N"])
            if block_size is None:
                block_size = int(row["blockSize"])

    return threads, speedup_simple, speedup_blocked, N, block_size


def main():
    parser = argparse.ArgumentParser(
        description="Plot Speedup vs Threads for matmul (simple vs blocked) from CSV."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to CSV file, e.g. results_matmul_N512_block64.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output image file name, e.g. matmul_speedup_N512_block64.png",
    )

    args = parser.parse_args()

    threads, s_simple, s_blocked, N, block_size = load_speedup_from_csv(args.csv_path)

    plt.style.use("default")

    plt.figure()

    # simple 版本
    plt.plot(threads, s_simple, marker="o", label="OpenMP simple")

    # blocked 版本
    plt.plot(threads, s_blocked, marker="s", label="OpenMP blocked")

    plt.xlabel("Threads (p)")
    plt.ylabel("Speedup $S_p$")
    plt.title(f"Matrix Multiplication Speedup vs Threads (N={N}, block={block_size})")
    plt.grid(True)
    plt.xticks(threads)
    plt.legend()

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved figure to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

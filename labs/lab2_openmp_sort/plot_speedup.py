import argparse
import csv

import matplotlib.pyplot as plt


def load_speedup_from_csv(path):
    threads = []
    speedups = []
    N = None
    threshold = None  # 可选字段

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 必须字段
            t = int(row["threads"])
            s = float(row["Speedup_Sp"])

            threads.append(t)
            speedups.append(s)

            if N is None:
                N = int(row["N"])

            # 如果有 threshold 字段就顺便记一下，没有就忽略
            if threshold is None and "threshold" in row and row["threshold"]:
                try:
                    threshold = int(row["threshold"])
                except ValueError:
                    threshold = None

    return threads, speedups, N, threshold


def main():
    parser = argparse.ArgumentParser(
        description="Plot Speedup vs Threads for OpenMP sort from CSV."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to CSV file, e.g. results_N1e6_threshold20000.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output image file name, e.g. speedup_N1e6.png",
    )

    args = parser.parse_args()

    threads, speedups, N, threshold = load_speedup_from_csv(args.csv_path)

    plt.style.use("default")
    plt.figure()

    plt.plot(threads, speedups, marker="o")

    plt.xlabel("Threads (p)")
    plt.ylabel("Speedup $S_p$")

    if threshold is not None:
        plt.title(f"OpenMP Sort Speedup vs Threads (N={N}, threshold={threshold})")
    else:
        plt.title(f"OpenMP Sort Speedup vs Threads (N={N})")

    plt.grid(True)
    plt.xticks(threads)

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved figure to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

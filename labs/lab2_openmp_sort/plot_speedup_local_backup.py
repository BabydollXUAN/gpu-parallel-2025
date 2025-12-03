import argparse
import csv

import matplotlib.pyplot as plt


def load_speedup_from_csv(path):
    threads = []
    speedups = []
    N = None
    threshold = None

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = int(row["threads"])
            s = float(row["Speedup_Sp"])
            threads.append(t)
            speedups.append(s)

            # 记录 N 和 threshold（用第一行就够）
            if N is None:
                N = int(row["N"])
            if threshold is None:
                threshold = int(row["threshold"])

    return threads, speedups, N, threshold


def main():
    parser = argparse.ArgumentParser(
        description="Plot Speedup vs Threads from a CSV file."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the CSV file (e.g., results_N1e6_threshold20000.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output image file name (e.g., speedup_N1e6.png). "
             "If not set, the plot will just show on screen.",
    )

    args = parser.parse_args()

    threads, speedups, N, threshold = load_speedup_from_csv(args.csv_path)

    # 使用默认风格
    plt.style.use("default")

    plt.figure()
    plt.plot(threads, speedups, marker="o")  # 不手动指定颜色

    plt.xlabel("Threads (p)")
    plt.ylabel("Speedup $S_p$")
    plt.title(f"Speedup vs Threads (N={N}, threshold={threshold})")
    plt.grid(True)

    # x 轴刻度设为整数线程数
    plt.xticks(threads)

    if args.output:
        plt.tight_layout()
        plt.savefig(args.output, dpi=150)
        print(f"Saved figure to {args.output}")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

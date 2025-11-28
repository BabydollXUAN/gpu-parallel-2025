import argparse
import csv

import matplotlib.pyplot as plt

def load_results(csv_path):
    Ns = []
    speedup_naive = []
    speedup_shared = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            Ns.append(int(row["N"]))
            speedup_naive.append(float(row["Speedup_naive"]))
            speedup_shared.append(float(row["Speedup_shared"]))
    return Ns, speedup_naive, speedup_shared

def main():
    parser = argparse.ArgumentParser(description="Plot CUDA reduction speedup vs N")
    parser.add_argument("csv_path", help="Path to results_reduce_vsN.csv")
    parser.add_argument("--output", default="figures/speedup_reduce_vsN.png",
                        help="Output figure path")
    args = parser.parse_args()

    Ns, speedup_naive, speedup_shared = load_results(args.csv_path)

    plt.figure()
    plt.plot(Ns, speedup_naive, marker="o", label="GPU naive")
    plt.plot(Ns, speedup_shared, marker="s", label="GPU shared (blocked)")

    plt.xlabel("Array size N")
    plt.ylabel("Speedup vs CPU")
    plt.title("CUDA Reduction Speedup vs N (blockSize=256)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    # 确保 figures 目录存在
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved figure to {args.output}")

if __name__ == "__main__":
    main()

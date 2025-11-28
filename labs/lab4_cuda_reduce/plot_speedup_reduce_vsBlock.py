import argparse
import csv
import os

import matplotlib.pyplot as plt

def load_results(csv_path):
    blocks = []
    speedup_shared2 = []
    speedup_naive = []
    speedup_shared = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            blocks.append(int(row["blockSize"]))
            speedup_shared2.append(float(row["Speedup_shared2"]))
            speedup_naive.append(float(row["Speedup_naive"]))
            speedup_shared.append(float(row["Speedup_shared"]))
    return blocks, speedup_shared2, speedup_naive, speedup_shared

def main():
    parser = argparse.ArgumentParser(description="Plot CUDA reduction speedup vs blockSize")
    parser.add_argument("csv_path", help="Path to results_reduce_vsBlock.csv")
    parser.add_argument(
        "--output",
        default="figures/speedup_reduce_vsBlock.png",
        help="Output figure path",
    )
    args = parser.parse_args()

    blocks, speedup_shared2, speedup_naive, speedup_shared = load_results(args.csv_path)

    plt.figure()
    plt.plot(blocks, speedup_naive, marker="o", label="GPU naive")
    plt.plot(blocks, speedup_shared, marker="s", label="GPU shared")
    plt.plot(blocks, speedup_shared2, marker="^", label="GPU shared-2elem")

    plt.xlabel("blockSize")
    plt.ylabel("Speedup vs CPU")
    plt.title("CUDA Reduction Speedup vs blockSize (N=1e7)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved figure to {args.output}")

if __name__ == "__main__":
    main()

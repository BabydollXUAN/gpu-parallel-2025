import csv
import argparse
import matplotlib.pyplot as plt

plt.style.use('default')

def load_results(path):
    Ns = []
    S_naive = []
    S_shared = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            Ns.append(int(row["N"]))
            S_naive.append(float(row["speedup_naive"]))
            S_shared.append(float(row["speedup_shared"]))
    return Ns, S_naive, S_shared

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="CSV file with N, speedup")
    parser.add_argument("--output", default="speedup_image_blur.png",
                        help="Output figure path")
    args = parser.parse_args()

    Ns, S_naive, S_shared = load_results(args.csv_path)

    plt.figure(figsize=(6,4))
    # 为了展示方便，这里只对 y 取对数，你也可以改成 plt.xscale("log", base=2)
    plt.xscale("log", base=2)
    plt.yscale("log")

    plt.plot(Ns, S_naive, marker="o", label="GPU naive vs CPU")
    plt.plot(Ns, S_shared, marker="s", label="GPU shared vs CPU")

    plt.xlabel("Image size N (N x N)")
    plt.ylabel("Speedup S_p (CPU time / GPU time)")
    plt.title("CUDA Image Blur: Speedup vs N")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"[+] Saved figure to {args.output}")

if __name__ == "__main__":
    main()

import csv
import argparse
import matplotlib.pyplot as plt

plt.style.use('default')

def load_results(path):
    # 结构：{block_size: {N: speedup_shared}}
    data = {}
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            N = int(row["N"])
            bs = int(row["block_size"])
            s_shared = float(row["speedup_shared"])
            data.setdefault(bs, {})[N] = s_shared
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="CSV file with N, block_size, speedup_shared")
    parser.add_argument("--output", default="speedup_image_blur_blocksize.png",
                        help="Output figure path")
    args = parser.parse_args()

    data = load_results(args.csv_path)

    # 我们只画 block_size=16 和 32 的 shared
    Ns = sorted(set(list(data.get(16, {}).keys()) + list(data.get(32, {}).keys())))

    S16 = [data.get(16, {}).get(N, None) for N in Ns]
    S32 = [data.get(32, {}).get(N, None) for N in Ns]

    plt.figure(figsize=(6,4))
    plt.xscale("log", base=2)
    plt.yscale("log")

    if any(v is not None for v in S16):
        plt.plot(Ns, S16, marker="o", label="Shared, block_size=16")
    if any(v is not None for v in S32):
        plt.plot(Ns, S32, marker="s", label="Shared, block_size=32")

    plt.xlabel("Image size N (N x N)")
    plt.ylabel("Speedup S_p (CPU / GPU shared)")
    plt.title("CUDA Image Blur: Effect of block size on shared-memory speedup")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"[+] Saved figure to {args.output}")

if __name__ == "__main__":
    main()

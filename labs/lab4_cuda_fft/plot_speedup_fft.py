import argparse
import csv
import os

import matplotlib.pyplot as plt

def load_results(csv_path):
    Ns = []
    cpu_times = []
    gpu_times = []
    speedups = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            Ns.append(int(row["N"]))
            cpu_times.append(float(row["T_cpu_ms"]))
            gpu_times.append(float(row["T_gpu_ms"]))
            speedups.append(float(row["Speedup"]))
    return Ns, cpu_times, gpu_times, speedups

def main():
    parser = argparse.ArgumentParser(description="Plot FFT Speedup vs N")
    parser.add_argument("csv_path", help="Path to results_fft_vsN.csv")
    parser.add_argument(
        "--output",
        default="figures/speedup_fft_vsN.png",
        help="Output figure path",
    )
    args = parser.parse_args()

    Ns, cpu_times, gpu_times, speedups = load_results(args.csv_path)

    plt.figure()
    plt.plot(Ns, speedups, marker="o")
    plt.xlabel("FFT size N")
    plt.ylabel("Speedup (CPU DFT / GPU cuFFT)")
    plt.title("CUDA FFT Speedup vs N")
    plt.grid(True, linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved figure to {args.output}")

if __name__ == "__main__":
    main()

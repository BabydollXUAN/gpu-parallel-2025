import os

import matplotlib
# 服务器上一般没有图形界面，用无窗口后端
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import imageio.v2 as iio
except ImportError:
    import imageio as iio  # 兼容老版本

def load_img(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found, 请先运行 bash run.sh 生成 .pgm 图像")
    img = iio.imread(path)
    return img

def main():
    input_path   = "input.pgm"
    cpu_path     = "ipm_cpu.pgm"
    gpu_path     = "ipm_gpu.pgm"

    img_in  = load_img(input_path)
    img_cpu = load_img(cpu_path)
    img_gpu = load_img(gpu_path)

    # 画三张图对比：原图 / CPU IPM / GPU IPM
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].set_title("Input (camera view)")
    axes[0].imshow(img_in, cmap="gray")
    axes[0].axis("off")

    axes[1].set_title("IPM - CPU")
    axes[1].imshow(img_cpu, cmap="gray")
    axes[1].axis("off")

    axes[2].set_title("IPM - GPU")
    axes[2].imshow(img_gpu, cmap="gray")
    axes[2].axis("off")

    plt.tight_layout()
    out_png = "ipm_compare.png"
    plt.savefig(out_png, dpi=200)
    print(f"[+] Saved comparison figure to {out_png}")

if __name__ == "__main__":
    main()

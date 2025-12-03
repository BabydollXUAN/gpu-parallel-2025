import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import imageio.v2 as iio
except ImportError:
    import imageio as iio

def load_gray(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found, 请先运行 bash run.sh 生成 IPM 图像")
    img = iio.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32)

def sobel_edge(img: np.ndarray) -> np.ndarray:
    # Sobel 核
    Kx = np.array([[ -1, 0, 1],
                   [ -2, 0, 2],
                   [ -1, 0, 1]], dtype=np.float32)
    Ky = np.array([[  1,  2,  1],
                   [  0,  0,  0],
                   [ -1, -2, -1]], dtype=np.float32)

    H, W = img.shape
    padded = np.pad(img, pad_width=1, mode='edge')
    Gx = np.zeros_like(img, dtype=np.float32)
    Gy = np.zeros_like(img, dtype=np.float32)

    for y in range(H):
        for x in range(W):
            region = padded[y:y+3, x:x+3]
            Gx[y, x] = np.sum(region * Kx)
            Gy[y, x] = np.sum(region * Ky)

    mag = np.sqrt(Gx**2 + Gy**2)

    mag = mag - mag.min()
    if mag.max() > 0:
        mag = mag / mag.max() * 255.0
    return mag.astype(np.uint8)

def main():
    ipm_gpu_path = "ipm_gpu.pgm"
    img_ipm = load_gray(ipm_gpu_path)

    edges = sobel_edge(img_ipm)

    out_pgm = "ipm_gpu_sobel.pgm"
    out_png = "ipm_gpu_sobel.png"

    iio.imwrite(out_pgm, edges)
    print(f"[+] Saved sobel result to {out_pgm}")

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("IPM - GPU (bird's-eye)")
    plt.imshow(img_ipm, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("IPM + Sobel edges")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"[+] Saved comparison figure to {out_png}")

if __name__ == "__main__":
    main()

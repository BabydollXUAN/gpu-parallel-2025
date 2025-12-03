import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2  # 新增：用 OpenCV 读图 & resize

def find_input_image():
    """
    优先使用 images/road.png，如果不存在再尝试 images/road.jpg / road.jpeg。
    """
    candidates = [
        os.path.join("images", "road.png"),
        os.path.join("images", "road.jpg"),
        os.path.join("images", "road.jpeg"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "未找到输入图像，请在 images/ 目录下准备 road.png 或 road.jpg / road.jpeg"
    )

def main():
    src = find_input_image()
    print(f"[+] Using input image: {src}")

    # 用 OpenCV 读彩色图，再转灰度
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"无法读取图像: {src}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 统一 resize 到 (640, 480)：宽 640，高 480
    target_w, target_h = 640, 480
    gray_resized = cv2.resize(
        gray, (target_w, target_h), interpolation=cv2.INTER_AREA
    )

    # OpenCV 已经是 uint8 的 [0,255]，直接写 PGM
    out_pgm = "input.pgm"
    # 用 imageio 写成 PGM（也可以用 cv2.imwrite）
    import imageio.v2 as iio
    iio.imwrite(out_pgm, gray_resized)
    print(f"[+] Saved grayscale PGM to {out_pgm}, shape={gray_resized.shape}")

    # 顺便输出一张 PNG 预览
    preview_png = "input_preview.png"
    plt.figure(figsize=(4, 4))
    plt.imshow(gray_resized, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(preview_png, dpi=200)
    print(f"[+] Saved preview figure to {preview_png}")

if __name__ == "__main__":
    main()


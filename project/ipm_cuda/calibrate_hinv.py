import cv2
import numpy as np

# 读入你真正喂给 C++ 的那张图
img = cv2.imread("input.pgm", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.pgm not found, 先用脚本生成它")

h, w = img.shape[:2]
print("Image shape:", w, "x", h)

# === 1) 手动指定 4 个路面角点（源图像上的点） ===
# 建议顺序：左下、右下、右上、左上（或者任意顺序，只要 src 和 dst 一致）
#
# 你可以先用下面这段可视化找坐标：
#   - 在窗口里移动鼠标，读坐标（右下角有 x,y）
#   - 或者自己用 matplotlib 标点
#
# 这里先给一个“示例写法”，请把数字改成你自己读出来的：
src_pts = np.float32([
    [220, 430],   # 左下
    [420, 430],   # 右下
    [360, 300],   # 右上
    [260, 300],   # 左上
])

# === 2) 目标鸟瞰平面上的 4 个点（矩形） ===
# 我们希望车道在 IPM 图里变成垂直的长矩形，比如横向放在中间：
dst_pts = np.float32([
    [220, 480],   # 左下
    [420, 480],   # 右下
    [420, 200],   # 右上
    [220, 200],   # 左上
])

# 3) 计算 src -> dst 的单应矩阵 H
H = cv2.getPerspectiveTransform(src_pts, dst_pts)

# 4) 反求 Hinv（输出要给 C++ 用：输出像素 -> 输入像素）
Hinv = np.linalg.inv(H)

print("Hinv (numpy):\n", Hinv)

# 5) 给一段 C++ 初始化用的 row-major 输出
print("\nCopy this into get_Hinv_host (row-major):")
flat = Hinv.reshape(-1)
for i, v in enumerate(flat):
    end = "\n" if i % 3 == 2 else " "
    print(f"{v:.7f}f,", end=end)

# 6) 验证一下效果：直接用 OpenCV 做一次 IPM，存到 ipm_cv.png
ipm_cv = cv2.warpPerspective(img, H, (w, h))  # 注意是 H，不是 Hinv
cv2.imwrite("ipm_cv.png", ipm_cv)
print("Saved ipm_cv.png for visual check.")

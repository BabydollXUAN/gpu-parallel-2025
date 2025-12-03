import cv2
import sys

# 1. 这里填你的真实图片文件名 (jpg/png 都可以)
# input_filename = "real_road.jpg"  # <--- 请修改这里！！！
input_filename = "images/road.jpg"

# 读取图片
img = cv2.imread(input_filename)
if img is None:
    print(f"找不到文件: {input_filename}")
    sys.exit(1)

# 2. 调整大小 (可选)
# 为了和你的 C++ 代码默认设置匹配，也为了计算快一点，
# 建议把图片统一缩放到 640x480 或者 1280x720
img = cv2.resize(img, (640, 480)) 

# 3. 转成灰度图
# C++ 代码只处理单通道灰度图，所以这里必须转
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4. 保存为 input.pgm
# OpenCV 会自动根据后缀名保存为 P5 格式 (Binary PGM)
cv2.imwrite("input.pgm", gray)

print(f"成功！已将 {input_filename} 转换为 input.pgm (640x480)")
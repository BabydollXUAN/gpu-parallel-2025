import cv2
import numpy as np

def get_projection_matrix(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    h, w = img.shape[:2]
    print(f"Image Size: {w}x{h}")
    
    # ---------------------------------------------------------
    # 交互式选点：请在弹出的窗口中，按【左上 -> 右上 -> 右下 -> 左下】的顺序
    # 点击车道线围成的梯形区域的四个角点。
    # ---------------------------------------------------------
    print("Please click 4 points in this order: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
    src_points = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(src_points) < 4:
                print(f"Clicked point: {x}, {y}")
                src_points.append([x, y])
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Select Points", img)

    cv2.imshow("Select Points", img)
    cv2.setMouseCallback("Select Points", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(src_points) != 4:
        print("Error: You must select exactly 4 points.")
        return

    src_pts = np.float32(src_points)

    # ---------------------------------------------------------
    # 定义目标点 (Destination Points)
    # 我们希望把选中的梯形变换成一个矩形。
    # 既然是鸟瞰图，我们假设输出图像大小和输入一样，或者自定义
    # 这里的设置决定了鸟瞰图的比例尺
    # ---------------------------------------------------------
    # 简单的策略：让变换后的区域占据整个图像的中间部分
    # 这里的顺序必须和点击顺序一致：左上，右上，右下，左下
    margin_x = w // 4
    margin_y = 0 
    dst_pts = np.float32([
        [margin_x, 0],       # 对应左上 (Top-Left)
        [w - margin_x, 0],   # 对应右上 (Top-Right)
        [w - margin_x, h],   # 对应右下 (Bottom-Right)
        [margin_x, h]        # 对应左下 (Bottom-Left)
    ])

    # ---------------------------------------------------------
    # 计算矩阵
    # C++ 代码逻辑是: Hinv * [u_out, v_out, 1] = [x_src, y_src, w]
    # 也就是是从 "目标图(Dst)" 映射回 "源图(Src)"
    # OpenCV 的 getPerspectiveTransform(src, dst) 计算的是 src -> dst
    # 所以我们需要反过来：getPerspectiveTransform(dst, src)
    # ---------------------------------------------------------
    H_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    print("\nSUCCESS! Paste the following code into your C++ file (get_Hinv_host function):\n")
    print("-" * 60)
    print("// Auto-generated Hinv matrix")
    print(f"Hinv[0] = {H_inv[0,0]:.8f}f; Hinv[1] = {H_inv[0,1]:.8f}f; Hinv[2] = {H_inv[0,2]:.8f}f;")
    print(f"Hinv[3] = {H_inv[1,0]:.8f}f; Hinv[4] = {H_inv[1,1]:.8f}f; Hinv[5] = {H_inv[1,2]:.8f}f;")
    print(f"Hinv[6] = {H_inv[2,0]:.8f}f; Hinv[7] = {H_inv[2,1]:.8f}f; Hinv[8] = {H_inv[2,2]:.8f}f;")
    print("-" * 60)
    
    # 验证一下效果 (Optional)
    # warped = cv2.warpPerspective(cv2.imread(image_path), np.linalg.inv(H_inv), (w, h))
    # cv2.imshow("Preview Result", warped)
    # cv2.waitKey(0)

# 替换成你的真实图片路径，如果是 pgm 也可以直接读取，只要 opencv 支持
# 建议先转成 jpg/png 方便调试，或者确保 opencv 能读 pgm
# get_projection_matrix("input_real.jpg") 
if __name__ == "__main__":
    # 如果你只有 input.pgm，请确保它存在
    import os
    if os.path.exists("input.pgm"):
        get_projection_matrix("input.pgm")
    else:
        print("Please verify input.pgm exists or change the path in the script.")
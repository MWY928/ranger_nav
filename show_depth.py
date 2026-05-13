from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. 读取 CSV 深度数据
# =========================
csv_path = Path(r"D:\depth_output\depth_logs\step50.csv")
#csv_path = Path(r"test_modules\test_results\bridge_depth_samples\depth_sample_1778491004401717424_meter.csv")

# 读取二维深度矩阵
depth = np.loadtxt(csv_path, delimiter=",", skiprows=1, encoding="utf-8-sig")

print("Depth shape:", depth.shape)
print("Min depth:", np.min(depth))
print("Max depth:", np.max(depth))

# =========================
# 2. 处理异常值
# =========================
# 将 NaN 替换为 0
depth = np.nan_to_num(depth)

# =========================
# 3. 深度归一化到 0~255
# =========================
depth_min = depth.min()
depth_max = depth.max()

depth_norm = (depth - depth_min) / (depth_max - depth_min)

# 转 uint8
depth_uint8 = (depth_norm * 255).astype(np.uint8)

# # =========================
# # 4. 保存灰度深度图
# # =========================
# cv2.imwrite("depth_gray.png", depth_uint8)

# print("Gray depth image saved as depth_gray.png")


# 5. 保存彩色深度图

depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

cv2.imwrite("depth_color.png", depth_color)

print("Color depth image saved as depth_color.png")

# 6. 显示结果

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Gray Depth")
plt.imshow(depth_uint8, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Color Depth")
plt.imshow(cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB))

plt.show()

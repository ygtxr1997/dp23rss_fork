import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# 示例的二维轨迹
trajectory = np.array([[0, 0], [1, 2], [2, 3], [3, 5], [4, 4]])
timestamps = np.linspace(0, 1, len(trajectory))  # 假设每个点有时间戳，范围在 0 到 1

# 提取 x 和 y 坐标
x = trajectory[:, 0]
y = trajectory[:, 1]

# 创建渐变颜色的线
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# 使用 LineCollection 实现颜色渐变
lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, 1))
lc.set_array(timestamps)  # 设置时间作为颜色映射的值
lc.set_linewidth(2)  # 线宽

# 绘图
plt.figure(figsize=(8, 6))
plt.gca().add_collection(lc)
plt.scatter(x, y, c=timestamps, cmap='viridis', edgecolor='k')  # 绘制节点，带颜色渐变
plt.colorbar(lc, label="Time")  # 添加颜色条
plt.title("2D Trajectory with Time-based Color Gradient")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.xlim(x.min() - 1, x.max() + 1)
plt.ylim(y.min() - 1, y.max() + 1)

# 保存为图像文件
plt.savefig("trajectory_with_gradient.png", dpi=300)
plt.show()

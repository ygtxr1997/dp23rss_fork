import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FixedLocator, FixedFormatter

# 设置论文风格
plt.style.use("seaborn-v0_8-paper")

# 数据
# [Orange]
# x = [0.24006866249832184, 0.15864046030938886, 0.19663101646946074]
# y = [0.1547, 0.2429, 0.2193]
# std = [0.0133, 0.0199, 0.0272]  # 误差条标准差
# [Size]
x = [0.12396835633530934, 0.05879708937306585, 0.06863101646946074]
y = [0.3623, 0.4898, 0.4641]
std = [0.0228, 0.0446, 0.0466]  # 误差条标准差
# y = [0.492, 0.5234, 0.5064]
# std = [0.031, 0.0212, 0.0307]
std = [s / 2 for s in std]

labels = ["Baseline", "HDFree", "HDFree (vis)"]
colors = ["blue", "red", "green"]  # 每个点不同颜色
markers = ["o", "*", "s"]  # 第二个点使用五角星 (*)

# 创建图像
plt.figure(figsize=(4, 4), dpi=300)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# 绘制散点 + 误差条
for i in range(len(x)):
    if i == 1:  # 第二个点（特殊样式：红色空心五角星）
        plt.scatter(x[i], y[i], color="none", edgecolors="red", s=200, marker="*", linewidths=2, label=labels[i])
    else:
        plt.scatter(x[i], y[i], color=colors[i], edgecolors="black", s=100, marker=markers[i], alpha=0.8, label=labels[i])

    # 添加误差条
    plt.errorbar(x[i], y[i], yerr=std[i], fmt="o", color=colors[i], capsize=5, capthick=1.5, elinewidth=1.5, alpha=0.8, ecolor=colors[i])

    # 标注数据点（避免与误差条重叠）
    offset_x = -0.005 if i == 0 else 0.005
    offset_y = 0.01 if i == 0 else 0.01
    plt.text(x[i] + offset_x, y[i] + offset_y, labels[i], fontsize=16, ha='right' if i == 0 else 'left', va='bottom')

# 定义箭头样式
arrow_style = dict(arrowstyle="-|>", color="red", lw=2, mutation_scale=15)

# 添加曲线箭头
arrow1 = FancyArrowPatch((x[0], y[0]), (x[2], y[2]), connectionstyle="arc3,rad=0.2", **arrow_style)
arrow2 = FancyArrowPatch((x[2], y[2]), (x[1], y[1]), connectionstyle="arc3,rad=0.2", **arrow_style)

plt.gca().add_patch(arrow1)
plt.gca().add_patch(arrow2)

# 设置坐标轴标签（LaTeX 方式格式化）
# plt.xlabel(r"$\mathrm{Wasserstein\ Distance}$", fontsize=14)
# plt.ylabel(r"$\mathrm{IoU}$", fontsize=14)

# 获取当前 x/y 轴刻度
x_ticks = np.linspace(min(x) * 1.05, max(x) * 0.95, num=4)
y_ticks = np.linspace(int(min(y) * 100) / 100., int(max(y) * 100) / 100., num=5)

# 先设置刻度，再强制格式化为两位小数
plt.xticks(x_ticks, [f"{tick:.2f}" for tick in x_ticks], fontsize=10)
plt.yticks(y_ticks, [f"{tick:.2f}" for tick in y_ticks], fontsize=10, rotation=90)

# 也可以使用 FixedLocator & FixedFormatter
plt.gca().xaxis.set_major_locator(FixedLocator(x_ticks))
plt.gca().xaxis.set_major_formatter(FixedFormatter([f"{tick:.2f}" for tick in x_ticks]))
plt.gca().yaxis.set_major_locator(FixedLocator(y_ticks))
plt.gca().yaxis.set_major_formatter(FixedFormatter([f"{tick:.2f}" for tick in y_ticks]))

# 调整刻度方向
# **让刻度方向朝内，并调整标签位置**
plt.tick_params(axis="x", direction="in", length=6, width=1.2, pad=-15)  # x 轴刻度朝内，调整 pad
plt.tick_params(axis="y", direction="in", length=6, width=1.2, pad=-20)  # y 轴刻度朝内，调整 pad

# **让刻度标签对齐**
for label in plt.gca().get_xticklabels():
    label.set_horizontalalignment('center')  # x 轴刻度标签居中
    label.set_verticalalignment('top')  # x 轴刻度标签往上对齐
for label in plt.gca().get_yticklabels():
    label.set_horizontalalignment('right')  # y 轴刻度标签靠右
    label.set_verticalalignment('center')  # y 轴刻度标签居中

plt.xlim(min(x) * 0.8, max(x) * 1.05)
plt.ylim(min(y) * 0.95, max(y) * 1.07)

# 添加网格（适度透明）
plt.grid(alpha=0.3, linestyle="--", zorder=0)

# 增加边框线条粗细
plt.gca().spines["top"].set_linewidth(1.2)
plt.gca().spines["right"].set_linewidth(1.2)
plt.gca().spines["left"].set_linewidth(1.2)
plt.gca().spines["bottom"].set_linewidth(1.2)

# 添加图例
# plt.legend(fontsize=12, loc="lower left", frameon=True)

# 显示图像
plt.show()

# 保存高清图
plt.savefig("data/pusht_eval_output/distance.png", dpi=300, bbox_inches="tight", pad_inches=0.05)

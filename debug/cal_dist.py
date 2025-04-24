from IPython.core.pylabtools import figsize
from scipy.stats import wasserstein_distance
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import ot


def batch_wasserstein_distance(matrix1, matrix2):
    # 假设 matrix1 和 matrix2 的形状为 (B, 3000, 2)
    distances_x = []
    distances_y = []
    for i in range(matrix1.shape[0]):  # 遍历每个分布
        dist_x = wasserstein_distance(matrix1[i, :, 0], matrix2[i, :, 0])
        dist_y = wasserstein_distance(matrix1[i, :, 1], matrix2[i, :, 1])
        distances_x.append(dist_x)  # 平均 Wasserstein 距离
        distances_y.append(dist_y)  # 平均 Wasserstein 距离
    all_mean = (np.array(distances_x).mean() + np.array(distances_y).mean()) / 2
    return all_mean, (np.array(distances_x).mean(), np.array(distances_y).mean())  # 返回形状为 (B,)


def batch_wasserstein_distance_2d(matrix1, matrix2):
    """
    计算批量二维 Wasserstein 距离（使用最优传输方法）。

    参数：
    - matrix1: 形状 (B, 3000, 2) 的 NumPy 数组，表示 B 组二维样本分布
    - matrix2: 形状 (B, 3000, 2) 的 NumPy 数组，表示 B 组二维样本分布

    返回：
    - mean_wasserstein_distance: 所有批次 Wasserstein 距离的均值 (float)
    - wasserstein_distances: 每个批次 Wasserstein 距离的列表 (NumPy 数组)
    """
    # 确保输入形状一致
    assert matrix1.shape == matrix2.shape, "两个矩阵的形状必须一致"
    print("Calculating Wasserstein distances... (this may take a while)")

    B, N, D = matrix1.shape  # B = 批量大小, N = 点数, D = 2 (二维)
    wasserstein_distances = []

    for i in range(B):
        # 计算欧几里得距离矩阵（成本矩阵）
        C = ot.dist(matrix1[i], matrix2[i], metric='euclidean')

        # 计算最优传输距离（Wasserstein-2 距离）
        a = np.ones(N) / N  # 概率分布权重（均匀）
        b = np.ones(N) / N
        # W2_distance = ot.emd2(a, b, C)  # 计算 Wasserstein 距离
        W2_distance = ot.sinkhorn2(a, b, C, reg=0.01, numItermax=50000)

        wasserstein_distances.append(W2_distance)

    wasserstein_distances = np.array(wasserstein_distances)
    mean_wasserstein_distance = wasserstein_distances.mean()

    return mean_wasserstein_distance, wasserstein_distances


def remove_zero_points(matrix):
    """
    移除轨迹中所有值为 (0,0) 的点。
    参数:
    - matrix: 形状 (3000, 2) 的轨迹数据
    返回:
    - 过滤后的轨迹点
    """
    mask = ~(matrix == [0, 0]).all(axis=1)  # 仅保留非 (0,0) 的点
    return matrix[mask]


def batch_wasserstein_distance_weighted(matrix1, matrix2, decay_factor=0.01):
    """
    计算批量轨迹的 Wasserstein-2 距离，并赋予初始点更高权重。

    参数：
    - matrix1: 形状 (B, 3000, 2) 的 NumPy 数组，表示 B 组轨迹
    - matrix2: 形状 (B, 3000, 2) 的 NumPy 数组，表示 B 组轨迹
    - decay_factor: 控制轨迹起始点权重（越大初始点权重越高）

    返回：
    - mean_wasserstein_distance: 所有批次 Wasserstein-2 距离的均值
    - wasserstein_distances: 每个批次的 Wasserstein-2 距离（数组）
    """
    assert matrix1.shape == matrix2.shape, "轨迹矩阵形状不匹配"
    print("Calculating Wasserstein distances... (this may take a while)")

    B, N, D = matrix1.shape  # B = 批量大小, N = 轨迹点数, D = 2 (二维)
    wasserstein_distances = []

    for i in range(B):
        # 计算欧几里得距离矩阵（成本矩阵）
        C = ot.dist(matrix1[i], matrix2[i], metric='euclidean')

        # 计算非均匀权重（初始点权重更大）
        a = np.exp(-decay_factor * np.arange(N))
        a /= a.sum()  # 归一化成概率分布

        b = np.exp(-decay_factor * np.arange(N))
        b /= b.sum()  # 归一化

        # 计算最优传输距离（Wasserstein-2 距离）
        # W2_distance = ot.emd2(a, b, C)  # 计算 Wasserstein 距离
        W2_distance = ot.sinkhorn2(a, b, C, reg=0.01, numItermax=10000)

        wasserstein_distances.append(W2_distance)

    wasserstein_distances = np.array(wasserstein_distances)
    mean_wasserstein_distance = wasserstein_distances.mean()

    return mean_wasserstein_distance, wasserstein_distances


while True:
    # 示例
    domain_shift = "size"  # 'orange', 'texture', 'size'
    matrix1 = np.load("data/pusht_eval_output/baseline.npy") / 512
    matrix2 = np.load(f"data/pusht_eval_output/{domain_shift}.npy") / 512
    matrix3 = np.load(f"data/pusht_eval_output/{domain_shift}_hdfree.npy") / 512
    matrix4 = np.load(f"data/pusht_eval_output/{domain_shift}_hdfree_vis.npy") / 512
    matrix5 = np.load("data/pusht_eval_output/light.npy") / 512

    # plt.figure(figsize=(6, 6))
    # print(matrix1.shape, matrix2.shape, matrix3.shape)

    # wasserstein_dist, _ = batch_wasserstein_distance_weighted(matrix1, matrix2)
    # x_max = wasserstein_dist * 1.02
    # y_max = wasserstein_dist * 1.02
    # plt.scatter(x=wasserstein_dist, y=wasserstein_dist, label="dist1")
    # plt.text(wasserstein_dist, wasserstein_dist, f'dist1', fontsize=8, ha='left', va='bottom')
    # print("Wasserstein Distance:", wasserstein_dist)

    wasserstein_dist, _ = batch_wasserstein_distance_weighted(matrix1, matrix3)
    plt.scatter(x=wasserstein_dist, y=wasserstein_dist, label="dist2")
    plt.text(wasserstein_dist, wasserstein_dist, f'dist2', fontsize=8, ha='left', va='bottom')
    print("Wasserstein Distance:", wasserstein_dist)

    # wasserstein_dist, _ = batch_wasserstein_distance_weighted(matrix1, matrix4)
    # plt.scatter(x=wasserstein_dist, y=wasserstein_dist, label="dist3")
    # plt.text(wasserstein_dist, wasserstein_dist, f'dist3', fontsize=8, ha='left', va='bottom')
    # print("Wasserstein Distance:", wasserstein_dist)


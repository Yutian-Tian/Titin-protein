"""
系统：N_samples条由2个domain串联而成的链，每一个domain的轮廓长度是相同的，但是能垒满足一个高斯分布
目的：通过高斯分布实现本构曲线的连续化
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize_scalar
from scipy.stats import truncnorm

# ==================== 参数设置 ====================
# 输出目录（请修改为实际路径）
output_dir = "/home/tyt/project/Single-chain/opt+R/Rand_xi/Helmholtz_Optimization_results/100chains" 
os.makedirs(output_dir, exist_ok=True)

# 能垒采样参数
N_samples = 100          # 样本数
mu = 5.0                # 高斯分布的均值
sigma = 5.0              # 标准差
delta = 4.9              # 截断范围 [μ-δ, μ+δ]

# 物理参数（与参考代码一致）
xi_f = 5.0              # 第一个domain的折叠态长度
k = 1.1                  # k = ξ_ui / ξ_fi
r_grids = 1000           # 拉伸长度r的网格点数

# ==================== 辅助函数 ====================
def contour_length_Lci(n_i, xi_fi):
    """轮廓长度 L_ci = ξ_fi + n_i (k-1) ξ_fi"""
    return xi_fi + n_i * (k - 1) * xi_fi

def energy_term_U(n_i, DeltaEi):
    """能量项 U(n) = ΔE n - ΔE cos(2π n)"""
    return DeltaEi * n_i - DeltaEi * np.cos(2 * np.pi * n_i)

def WLC_free_energy(x_i, L_ci):
    """Marko–Siggia 自由能（WLC模型）"""
    if x_i >= 1.0 - 1e-12:
        return 1e300
    return 0.25 * L_ci * (x_i**2 * (3.0 - 2.0 * x_i) / (1.0 - x_i))

def single_domain_free_energy(r_i, n_i, DeltaEi, xi_fi):
    """单个domain的自由能"""
    L_ci = contour_length_Lci(n_i, xi_fi)
    if r_i < 0 or r_i > L_ci:
        return 1e300
    x_i = r_i / L_ci
    F_wlc = WLC_free_energy(x_i, L_ci)
    Ui = energy_term_U(n_i, DeltaEi)
    return F_wlc + Ui

def free_energy_2_domain(r, r1, n1, n2, DeltaE1, DeltaE2):
    """两个domain的总自由能"""
    energy1 = single_domain_free_energy(r1, n1, DeltaE1, xi_f)
    r2 = r - r1
    energy2 = single_domain_free_energy(r2, n2, DeltaE2, xi_f)
    total = energy1 + energy2
    return total if np.isfinite(total) else 1e300

def optimize_single_point(r, DeltaE1, DeltaE2,
                          init_points=15, refine_levels=6, refine_points=15, tol=1e-6):
    """
    对给定的拉伸长度r，优化自由能，返回 [r, r1_opt, n1_opt, n2_opt]
    """
    n1_min, n1_max = 0.0, 1.0
    n2_min, n2_max = 0.0, 1.0
    best_r1 = None
    best_n1 = None
    best_n2 = None
    best_F = float('inf')

    for level in range(refine_levels + 1):
        N = init_points if level == 0 else refine_points
        n1_grid = np.linspace(n1_min, n1_max, N)
        n2_grid = np.linspace(n2_min, n2_max, N)

        level_best_F = float('inf')
        level_best_r1 = None
        level_best_n1 = None
        level_best_n2 = None

        for n1 in n1_grid:
            for n2 in n2_grid:
                Lc1 = contour_length_Lci(n1, xi_f)
                Lc2 = contour_length_Lci(n2, xi_f)
                r1_lower = max(0.0, r - min(r, Lc2))
                r1_upper = min(Lc1, r)
                if r1_lower > r1_upper:
                    continue

                # 对r1进行一维优化
                res = minimize_scalar(
                    lambda r1: free_energy_2_domain(r, r1, n1, n2, DeltaE1, DeltaE2),
                    bounds=(r1_lower, r1_upper),
                    method='bounded',
                    options={'xatol': 1e-8, 'maxiter': 100}
                )
                if res.success and res.fun < level_best_F:
                    level_best_F = res.fun
                    level_best_r1 = res.x
                    level_best_n1 = n1
                    level_best_n2 = n2

        if level_best_F == float('inf'):
            return np.array([r, np.nan, np.nan, np.nan])

        if level_best_F < best_F:
            best_F = level_best_F
            best_r1 = level_best_r1
            best_n1 = level_best_n1
            best_n2 = level_best_n2

        if level == refine_levels:
            break

        # 确定下一级细化范围
        idx_n1 = np.argmin(np.abs(n1_grid - best_n1))
        idx_n2 = np.argmin(np.abs(n2_grid - best_n2))
        left_n1 = n1_grid[max(0, idx_n1 - 1)]
        right_n1 = n1_grid[min(N - 1, idx_n1 + 1)]
        left_n2 = n2_grid[max(0, idx_n2 - 1)]
        right_n2 = n2_grid[min(N - 1, idx_n2 + 1)]
        n1_min, n1_max = left_n1, right_n1
        n2_min, n2_max = left_n2, right_n2

        if (n1_max - n1_min < tol) and (n2_max - n2_min < tol):
            break

    return np.array([r, best_r1, best_n1, best_n2])

# ==================== 主程序 ====================
def main():
    # ---------- Step 1: 采样能垒 ----------
    # 截断正态分布采样
    low, high = (mu - delta - mu) / sigma, (mu + delta - mu) / sigma
    DeltaE1_samples = truncnorm.rvs(low, high, loc=mu, scale=sigma, size=N_samples)
    DeltaE2_samples = truncnorm.rvs(low, high, loc=mu, scale=sigma, size=N_samples)

    # 保存为 energy.csv 到指定路径
    energy_path = os.path.join(output_dir, 'energy.csv')
    df_energy = pd.DataFrame({
        'group': np.arange(1, N_samples + 1),
        'DeltaE1': DeltaE1_samples,
        'DeltaE2': DeltaE2_samples
    })
    df_energy.to_csv(energy_path, index=False)
    print(f"Step 1 完成：能垒采样已保存至 {energy_path}")

    # ---------- Step 2: 准备r网格 ----------
    r_max = contour_length_Lci(1, xi_f) + contour_length_Lci(1, xi_f)
    r_vals = np.linspace(0, 0.95 * r_max, r_grids)

    # ---------- Step 3: 对每组能垒进行优化 ----------
    # 用于存储结果的列表，形状 (N_samples, len(r_vals))
    all_r1 = []
    all_n1 = []
    all_n2 = []

    for i in range(N_samples):
        print(f"正在处理第 {i+1}/{N_samples} 组: ΔE1={DeltaE1_samples[i]:.3f}, ΔE2={DeltaE2_samples[i]:.3f}")
        group_r1 = []
        group_n1 = []
        group_n2 = []
        for r in r_vals:
            res = optimize_single_point(r, DeltaE1_samples[i], DeltaE2_samples[i])
            group_r1.append(res[1])
            group_n1.append(res[2])
            group_n2.append(res[3])
        all_r1.append(group_r1)
        all_n1.append(group_n1)
        all_n2.append(group_n2)

    # 转换为数组并转置：每行对应一个r值，每列对应一个样本
    all_r1 = np.array(all_r1).T   # shape (len(r_vals), N_samples)
    all_n1 = np.array(all_n1).T
    all_n2 = np.array(all_n2).T

    # ---------- Step 4: 保存结果到指定路径（高效方式，避免碎片化警告）----------
    # 构建 r1 DataFrame
    data_r1 = {'r': r_vals}
    for i in range(N_samples):
        data_r1[f'chain_{i+1}'] = all_r1[:, i]
    df_r1 = pd.DataFrame(data_r1)

    # 构建 n1 DataFrame
    data_n1 = {'r': r_vals}
    for i in range(N_samples):
        data_n1[f'chain_{i+1}'] = all_n1[:, i]
    df_n1 = pd.DataFrame(data_n1)

    # 构建 n2 DataFrame
    data_n2 = {'r': r_vals}
    for i in range(N_samples):
        data_n2[f'chain_{i+1}'] = all_n2[:, i]
    df_n2 = pd.DataFrame(data_n2)

    # 保存文件
    r1_path = os.path.join(output_dir, 'r1.csv')
    n1_path = os.path.join(output_dir, 'n1.csv')
    n2_path = os.path.join(output_dir, 'n2.csv')
    df_r1.to_csv(r1_path, index=False)
    df_n1.to_csv(n1_path, index=False)
    df_n2.to_csv(n2_path, index=False)

    print(f"Step 3 完成：所有结果已保存至 {output_dir} ")

if __name__ == "__main__":
    main()
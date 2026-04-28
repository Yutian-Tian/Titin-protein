"""
目的：探究不同参数下的domain的解折叠顺序，参数包括xi_f1,xi_f2,E1,E2
系统：包含2个domain
主题：约束下优化系统的Helmholtz自由能
"""

import numpy as np
import pandas as pd
import os
import sys
from scipy.optimize import minimize_scalar

# 参数设置
xi_f1 = 5.0    # 第一个domain的折叠态长度
k = 2.0        # k = xi_ui/xi_fi
alpha = 0.5    # alpha = delta_Ei/xi_fi**2 (将被循环覆盖)
beta = 1.0     # beta = xi_f2/xi_f1       (将被循环覆盖)
E0 = 1.0

# 优化参数
r_grids = 1000

# 设置存储路径
save_path = "/home/tyt/project/Single-chain/opt+R/Rand_xi/Helmholtz_Optimization_results/2-domain_results"
os.makedirs(save_path, exist_ok=True)

def energy_term_U(n_i, DeltaEi):
    """
    能量项: U(n_i) = ΔE_i n_i - ΔE_i cos(2π n_i)
    """
    return DeltaEi * n_i - DeltaEi * np.cos(2 * np.pi * n_i)

def contour_length_Lci(n_i, xi_fi):
    """
    轮廓长度: L_{ci}(n_i) = ξ_fi + n_i (k - 1)ξ_fi
    """
    return xi_fi + n_i * (k - 1) * xi_fi

def end_to_end_factor_x_i(r_i, n_i, xi_fi):
    """
    端到端因子: x_i(r_i, n_i) = r_i / L_{ci}(n_i)
    """
    L_ci = contour_length_Lci(n_i, xi_fi)
    return r_i / L_ci

def WLC_free_energy(x_i, L_ci):
    if x_i >= 1.0 - 1e-12:          # 更保守的阈值
        return 1e300                 # 极大有限值
    return 0.25 * L_ci * (x_i**2 * (3.0 - 2.0 * x_i) / (1.0 - x_i))

def single_domain_free_energy(r_i, n_i, DeltaEi, xi_fi):
    L_ci = contour_length_Lci(n_i, xi_fi)
    if r_i < 0 or r_i > L_ci:
        return 1e300
    x_i = r_i / L_ci
    F_wlc = WLC_free_energy(x_i, L_ci)
    Ui = energy_term_U(n_i, DeltaEi)
    return F_wlc + Ui

def free_energy_2_domain(r, r1, n1, n2):
    # 第1个domain的能量
    xi_f2 = beta * xi_f1
    delta_E1 = E0 + alpha*E0*(xi_f1 - xi_f2)
    energy1 = single_domain_free_energy(r1, n1, delta_E1, xi_f1)
    # 第2个domain的能量
    r2 = r - r1
    delta_E2 = E0
    energy2 = single_domain_free_energy(r2, n2, delta_E2, xi_f2)
    total = energy1 + energy2
    if not np.isfinite(total):
        return 1e300
    return total

def Optimize_single_point(r, init_points=15, refine_levels=6, refine_points=15, tol=1e-6):
    """
    使用粗网格 + 局部细化优化给定拉伸 r 下的自由能。
    返回 [r, r1_opt, n1_opt, n2_opt]
    """
    n1_min, n1_max = 0.0, 1.0
    n2_min, n2_max = 0.0, 1.0
    best_r1 = None
    best_n1 = None
    best_n2 = None
    best_F = float('inf')

    for level in range(refine_levels + 1):
        # 当前层网格点数
        N = init_points if level == 0 else refine_points
        n1_grid = np.linspace(n1_min, n1_max, N)
        n2_grid = np.linspace(n2_min, n2_max, N)

        level_best_F = float('inf')
        level_best_r1 = None
        level_best_n1 = None
        level_best_n2 = None

        # 遍历当前网格
        for n1 in n1_grid:
            for n2 in n2_grid:
                Lc1 = contour_length_Lci(n1, xi_f1)
                Lc2 = contour_length_Lci(n2, beta * xi_f1)
                r1_lower = max(0.0, r - min(r, Lc2))
                r1_upper = min(Lc1, r)
                if r1_lower > r1_upper:
                    continue  # 不可行

                # 对 r1 进行一维凸优化
                res = minimize_scalar(
                    lambda r1: free_energy_2_domain(r, r1, n1, n2),
                    bounds=(r1_lower, r1_upper),
                    method='bounded',
                    options={'xatol': 1e-8, 'maxiter': 100}
                )
                if res.success and res.fun < level_best_F:
                    level_best_F = res.fun
                    level_best_r1 = res.x
                    level_best_n1 = n1
                    level_best_n2 = n2

        # 如果当前层无可行点，直接返回 NaN
        if level_best_F == float('inf'):
            return np.array([r, np.nan, np.nan, np.nan])

        # 更新全局最优
        if level_best_F < best_F:
            best_F = level_best_F
            best_r1 = level_best_r1
            best_n1 = level_best_n1
            best_n2 = level_best_n2

        # 最后一级细化结束
        if level == refine_levels:
            break

        # 确定下一级细化的范围：以当前最优为中心，取相邻网格点间距
        idx_n1 = np.argmin(np.abs(n1_grid - best_n1))
        idx_n2 = np.argmin(np.abs(n2_grid - best_n2))

        # 获取左右邻居（边界处理）
        left_n1 = n1_grid[max(0, idx_n1-1)]
        right_n1 = n1_grid[min(N-1, idx_n1+1)]
        left_n2 = n2_grid[max(0, idx_n2-1)]
        right_n2 = n2_grid[min(N-1, idx_n2+1)]

        n1_min, n1_max = left_n1, right_n1
        n2_min, n2_max = left_n2, right_n2

        # 检查是否达到精度
        if (n1_max - n1_min < tol) and (n2_max - n2_min < tol):
            break

    return np.array([r, best_r1, best_n1, best_n2])


def main():
    # 参数扫描范围
    alpha_values = np.arange(0.1, 0.5, 0.1)   # 0.5, 1.0, ..., 5.0
    beta_values  = np.arange(0.1, 1.1, 0.2)   # 0.2,...,1.0

    total_combinations = len(alpha_values) * len(beta_values)
    combination_counter = 0

    # 声明将修改全局变量 alpha, beta
    global alpha, beta

    for alpha_val in alpha_values:
        for beta_val in beta_values:
            combination_counter += 1
            print(f"\n处理组合 {combination_counter}/{total_combinations}: alpha={alpha_val:.1f}, beta={beta_val:.1f}")

            # 更新全局参数
            alpha = alpha_val
            beta  = beta_val

            # 重新计算当前参数下的最大拉伸
            r_max = contour_length_Lci(1, xi_f1) + contour_length_Lci(1, beta * xi_f1)
            r_vals = np.linspace(0, 0.95 * r_max, r_grids)

            results = []
            for i, r in enumerate(r_vals):
                opt_result = Optimize_single_point(r)
                r_val, r1_opt, n1_opt, n2_opt = opt_result
                r2_opt = r_val - r1_opt if not np.isnan(r1_opt) else np.nan
                results.append([r_val, r1_opt, r2_opt, n1_opt, n2_opt])

                # 每处理10%的点输出一次进度
                if (i + 1) % (r_grids // 10) == 0:
                    print(f"   已处理 {i+1} / {len(r_vals)} 个点")

            # 保存为CSV文件
            df = pd.DataFrame(results, columns=['r', 'r1', 'r2', 'n1', 'n2'])
            filename = f"alpha_{alpha_val:.1f}_beta_{beta_val:.1f}.csv"
            csv_filename = os.path.join(save_path, filename)
            df.to_csv(csv_filename, index=False)
            print(f"   结果已保存至: {csv_filename}")

if __name__ == "__main__":
    main()
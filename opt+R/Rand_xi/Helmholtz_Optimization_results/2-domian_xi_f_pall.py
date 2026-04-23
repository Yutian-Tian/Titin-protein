"""
系统：N_samples条由2个domain串联而成的链，每个domain的轮廓长度由xi_f决定，
能垒由xi_f通过ΔE = 1 + 4*(xi_f - μ + δ)计算得到。
目的：通过高斯分布采样实现本构曲线的连续化。
修改：每条链独立构建r网格，结果按列存储（每列对应一条链）。
并行化版本：使用multiprocessing并行处理各条链。
"""

import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize_scalar
from scipy.stats import truncnorm
import multiprocessing as mp
from functools import partial

# ==================== 参数设置 ====================
output_dir = "/home/tyt/project/Single-chain/opt+R/Rand_xi/Helmholtz_Optimization_results/2-domain_xi_f_pall_results"
os.makedirs(output_dir, exist_ok=True)

# 采样参数
N_samples = 100          # 样本数（链的条数）
Num = 2                  # 每条链包含的domain数
mu = 5.0                 # xi_f 高斯分布的均值
sigma = 1.0              # xi_f 标准差
delta = 3.0              # 截断范围 [μ-δ, μ+δ]

# 物理参数
alpha = 4.0              # xi_u = alpha * xi_f
r_grids = 1000           # 每条链独立使用的拉伸长度网格点数

# ==================== 辅助函数（与原始代码相同） ====================
def contour_length_Lci(n_i, xi_fi):
    """轮廓长度 L_ci = xi_fi + n_i * (alpha-1) * xi_fi"""
    return xi_fi + n_i * (alpha - 1) * xi_fi

def energy_term_U(n_i, DeltaEi):
    """能量项 U(n) = ΔE n - ΔE cos(2π n)"""
    return DeltaEi * n_i - DeltaEi * np.cos(2 * np.pi * n_i)

def WLC_free_energy(x_i, L_ci):
    """Marko–Siggia 自由能（WLC模型）"""
    if x_i >= 1.0:
        return 1e300
    else:
        return 0.25 * L_ci * (x_i**2 * (3.0 - 2.0 * x_i) / (1.0 - x_i))

def single_domain_free_energy(r_i, n_i, DeltaEi, xi_fi):
    """单个domain的自由能"""
    L_ci = contour_length_Lci(n_i, xi_fi)
    if r_i < 0 or r_i >= L_ci:
        return 1e300
    x_i = r_i / L_ci
    F_wlc = WLC_free_energy(x_i, L_ci)
    Ui = energy_term_U(n_i, DeltaEi)
    return F_wlc + Ui

def free_energy_2_domain(r, r1, n1, n2, DeltaE1, DeltaE2, xi_f1, xi_f2):
    """两个domain的总自由能"""
    energy1 = single_domain_free_energy(r1, n1, DeltaE1, xi_f1)
    r2 = r - r1
    energy2 = single_domain_free_energy(r2, n2, DeltaE2, xi_f2)
    total = energy1 + energy2
    return total if np.isfinite(total) else 1e300

def optimize_single_point(r, DeltaE1, DeltaE2, xi_f1, xi_f2,
                          init_points=30, refine_levels=6, refine_points=30, tol=1e-6):
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
                Lc1 = contour_length_Lci(n1, xi_f1)
                Lc2 = contour_length_Lci(n2, xi_f2)
                r1_lower = max(0.0, r - min(r, Lc2))
                r1_upper = min(Lc1, r)
                if r1_lower > r1_upper:
                    continue

                # 对r1进行一维优化
                res = minimize_scalar(
                    lambda r1: free_energy_2_domain(r, r1, n1, n2, DeltaE1, DeltaE2, xi_f1, xi_f2),
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

# ==================== 并行处理单条链的函数 ====================
def process_chain(chain_idx, xf1, xf2, dE1, dE2, alpha, r_grids):
    """
    处理单条链：构建r网格，对每个r进行优化，返回该链的所有结果。
    返回: (r1_array, r2_array, n1_array, n2_array)
    """
    # 当前链的最大总轮廓长度（n1=n2=1）
    Lc1_max = contour_length_Lci(1.0, xf1)
    Lc2_max = contour_length_Lci(1.0, xf2)
    total_max = Lc1_max + Lc2_max
    r_max = 0.95 * total_max   # 留一点余量

    # 独立构建 r 网格（从0到r_max，均匀分布 r_grids 个点）
    r_vals = np.linspace(0, r_max, r_grids)

    cur_r1 = []
    cur_n1 = []
    cur_n2 = []
    for r in r_vals:
        res = optimize_single_point(r, dE1, dE2, xf1, xf2)
        cur_r1.append(res[1])
        cur_n1.append(res[2])
        cur_n2.append(res[3])
    cur_r1 = np.array(cur_r1)
    cur_n1 = np.array(cur_n1)
    cur_n2 = np.array(cur_n2)
    cur_r2 = r_vals - cur_r1   # r2 = r - r1

    return cur_r1, cur_r2, cur_n1, cur_n2

# ==================== 主程序 ====================
def main():
    # ---------- Step 1: 采样 xi_f ----------
    low, high = (mu - delta - mu) / sigma, (mu + delta - mu) / sigma
    xi_f1_samples = truncnorm.rvs(low, high, loc=mu, scale=sigma, size=N_samples)
    xi_f2_samples = truncnorm.rvs(low, high, loc=mu, scale=sigma, size=N_samples)

    # 保存 xi_f.csv
    xi_f_path = os.path.join(output_dir, 'xi_f.csv')
    df_xi_f = pd.DataFrame({
        'group': np.arange(1, N_samples + 1),
        'xi_f1': xi_f1_samples,
        'xi_f2': xi_f2_samples
    })
    df_xi_f.to_csv(xi_f_path, index=False)
    print(f"Step 1 完成：xi_f 采样已保存至 {xi_f_path}")

    # ---------- Step 2: 根据 xi_f 计算 ΔE ----------
    DeltaE1_samples = 1 + 4 * (xi_f1_samples - mu + delta)
    DeltaE2_samples = 1 + 4 * (xi_f2_samples - mu + delta)

    energy_path = os.path.join(output_dir, 'DeltaE.csv')
    df_energy = pd.DataFrame({
        'group': np.arange(1, N_samples + 1),
        'DeltaE1': DeltaE1_samples,
        'DeltaE2': DeltaE2_samples
    })
    df_energy.to_csv(energy_path, index=False)
    print(f"Step 2 完成：ΔE 已保存至 {energy_path}")

    # ---------- Step 3 & 4: 并行处理所有链 ----------
    # 准备参数列表: (chain_idx, xf1, xf2, dE1, dE2, alpha, r_grids)
    args_list = [(i, xi_f1_samples[i], xi_f2_samples[i],
                  DeltaE1_samples[i], DeltaE2_samples[i],
                  alpha, r_grids) for i in range(N_samples)]

    # 使用进程池并行计算
    n_cores = mp.cpu_count()
    print(f"启动进程池，使用 {n_cores} 个 CPU 核心并行处理 {N_samples} 条链...")

    with mp.Pool(processes=n_cores) as pool:
        # starmap 可以解包参数元组
        results = pool.starmap(process_chain, args_list)

    # 解包结果
    all_r1 = []
    all_r2 = []
    all_n1 = []
    all_n2 = []
    for r1, r2, n1, n2 in results:
        all_r1.append(r1)
        all_r2.append(r2)
        all_n1.append(n1)
        all_n2.append(n2)

    print(f"并行计算完成，共处理 {len(results)} 条链。")

    # ---------- Step 5: 保存结果（每列一条链） ----------
    df_r1 = pd.DataFrame({f'chain_{i+1}': all_r1[i] for i in range(N_samples)})
    df_r2 = pd.DataFrame({f'chain_{i+1}': all_r2[i] for i in range(N_samples)})
    df_n1 = pd.DataFrame({f'chain_{i+1}': all_n1[i] for i in range(N_samples)})
    df_n2 = pd.DataFrame({f'chain_{i+1}': all_n2[i] for i in range(N_samples)})

    r1_path = os.path.join(output_dir, 'r1_values.csv')
    r2_path = os.path.join(output_dir, 'r2_values.csv')
    n1_path = os.path.join(output_dir, 'n1_values.csv')
    n2_path = os.path.join(output_dir, 'n2_values.csv')

    df_r1.to_csv(r1_path, index=False)
    df_r2.to_csv(r2_path, index=False)
    df_n1.to_csv(n1_path, index=False)
    df_n2.to_csv(n2_path, index=False)

    print(f"\n所有结果已保存至目录：{output_dir}")
    print(f"  - xi_f.csv        : 采样的 xi_f 值")
    print(f"  - DeltaE.csv      : 计算得到的 ΔE 值")
    print(f"  - r1_values.csv   : 每条链的 r1 (每列一条链，共 {r_grids} 行)")
    print(f"  - r2_values.csv   : 每条链的 r2")
    print(f"  - n1_values.csv   : 每条链的 n1")
    print(f"  - n2_values.csv   : 每条链的 n2")

if __name__ == "__main__":
    # 设置多进程启动方式（Linux下默认fork，Windows下需spawn，这里不强制，使用默认）
    # 确保在Windows下也能正常运行
    mp.freeze_support()
    main()
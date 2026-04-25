"""
系统：N_samples条由2个domain串联而成的链，每个domain的轮廓长度由xi_f决定，
能垒由xi_f通过ΔE = 1 + 4*(xi_f - μ + δ)计算得到。
目的：通过高斯分布采样实现本构曲线的连续化。
修改：每条链独立构建r网格，结果按列存储（每列对应一条链）。
"""

import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize_scalar
from scipy.stats import truncnorm

# ==================== 参数设置 ====================
output_dir = "/home/tyt/project/Single-chain/opt+R/Rand_xi/Helmholtz_Optimization_results/2-domain_xi_f_sampling_results"
os.makedirs(output_dir, exist_ok=True)

# 采样参数
N_samples = 100          # 样本数（链的条数）【调试时减小，正式计算可改回100】
Num = 2                  # 每条链包含的domain数
mu = 10.0                 # xi_f 高斯分布的均值
sigma = 3.0              # xi_f 标准差
delta = 3.0              # 截断范围 [μ-δ, μ+δ]

# 物理参数
alpha = 4.0              # xi_u = alpha * xi_f
r_grids = 200            # 每条链独立使用的拉伸长度网格点数【调试时减小】

# 优化参数
init_points = 30         # 初始粗网格点数
refine_levels = 4        # 细化层数（原为6，降低计算量）
refine_points = 20       # 每层细化时的网格点数
tol = 1e-6               # 细化收敛容差

# 数值容差
EPS_BOUND = 1e-12        # 边界零长度容差

# ==================== 辅助函数 ====================
def contour_length_Lci(n_i, xi_fi):
    """轮廓长度 L_ci = xi_fi + n_i * (alpha-1) * xi_fi"""
    return xi_fi + n_i * (alpha - 1) * xi_fi

def energy_term_U(n_i, DeltaEi):
    """能量项 U(n) = ΔE n - ΔE cos(2π n)"""
    return DeltaEi * n_i - DeltaEi * np.cos(2 * np.pi * n_i)

def WLC_free_energy(x_i, L_ci):
    """Marko–Siggia 自由能（WLC模型），添加数值稳定性处理"""
    if x_i >= 0.999999:   # 避免分母过小引起溢出
        return 1e100
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
                          init_points=30, refine_levels=4,
                          refine_points=20, tol=1e-6):
    """
    对给定的拉伸长度r，优化自由能，返回 [r, r1_opt, n1_opt, n2_opt]
    增加边界检查和异常处理，避免因退化区间崩溃。
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

                # 检查区间是否退化（长度过小）
                if r1_upper - r1_lower < EPS_BOUND:
                    # 直接取中点计算自由能
                    r1_mid = (r1_lower + r1_upper) * 0.5
                    F_val = free_energy_2_domain(r, r1_mid, n1, n2, DeltaE1, DeltaE2, xi_f1, xi_f2)
                    if F_val < level_best_F:
                        level_best_F = F_val
                        level_best_r1 = r1_mid
                        level_best_n1 = n1
                        level_best_n2 = n2
                    continue

                # 正常区间：调用 minimize_scalar
                try:
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
                except Exception as e:
                    # 优化失败时跳过该 (n1,n2) 组合
                    continue

        # 若当前层无任何可行解，终止细化并返回已有最优（可能为inf）
        if level_best_F == float('inf'):
            # 若从未找到任何可行解，返回全nan
            if best_F == float('inf'):
                return np.array([r, np.nan, np.nan, np.nan])
            else:
                break

        # 更新全局最优
        if level_best_F < best_F:
            best_F = level_best_F
            best_r1 = level_best_r1
            best_n1 = level_best_n1
            best_n2 = level_best_n2

        # 如果是最后一层，不再细化
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
    DeltaE1_samples = 5 + 4 * (xi_f1_samples - mu + delta)
    DeltaE2_samples = 5 + 4 * (xi_f2_samples - mu + delta)

    energy_path = os.path.join(output_dir, 'DeltaE.csv')
    df_energy = pd.DataFrame({
        'group': np.arange(1, N_samples + 1),
        'DeltaE1': DeltaE1_samples,
        'DeltaE2': DeltaE2_samples
    })
    df_energy.to_csv(energy_path, index=False)
    print(f"Step 2 完成：ΔE 已保存至 {energy_path}")

    # ---------- Step 3 & 4: 对每条链独立构建网格并优化 ----------
    all_r1 = []   # 每条链的 r1 值数组
    all_r2 = []   # 每条链的 r2 值数组
    all_n1 = []   # 每条链的 n1 值数组
    all_n2 = []   # 每条链的 n2 值数组
    all_r_vals = []   # 保存每条链的 r 网格（用于后续绘图）

    for i in range(N_samples):
        xf1 = xi_f1_samples[i]
        xf2 = xi_f2_samples[i]
        dE1 = DeltaE1_samples[i]
        dE2 = DeltaE2_samples[i]

        # 当前链的最大总轮廓长度（n1=n2=1）
        Lc1_max = contour_length_Lci(1.0, xf1)
        Lc2_max = contour_length_Lci(1.0, xf2)
        total_max = Lc1_max + Lc2_max
        r_max = 0.95 * total_max   # 留一点余量

        # 独立构建 r 网格
        r_vals = np.linspace(0, r_max, r_grids)

        cur_r1 = []
        cur_n1 = []
        cur_n2 = []

        for r in r_vals:
            res = optimize_single_point(r, dE1, dE2, xf1, xf2,
                                        init_points=init_points,
                                        refine_levels=refine_levels,
                                        refine_points=refine_points,
                                        tol=tol)
            cur_r1.append(res[1])
            cur_n1.append(res[2])
            cur_n2.append(res[3])

        cur_r1 = np.array(cur_r1)
        cur_n1 = np.array(cur_n1)
        cur_n2 = np.array(cur_n2)
        cur_r2 = r_vals - cur_r1

        all_r1.append(cur_r1)
        all_r2.append(cur_r2)
        all_n1.append(cur_n1)
        all_n2.append(cur_n2)
        all_r_vals.append(r_vals)

        print(f"已完成第 {i+1}/{N_samples} 条链: xi_f1={xf1:.3f}, xi_f2={xf2:.3f}, r_max={r_max:.3f}")

    # ---------- Step 5: 保存结果（每列一条链） ----------
    # 构建 DataFrame: 行对应网格点（所有链点数相同），列对应链编号
    df_r1 = pd.DataFrame({f'chain_{i+1}': all_r1[i] for i in range(N_samples)})
    df_r2 = pd.DataFrame({f'chain_{i+1}': all_r2[i] for i in range(N_samples)})
    df_n1 = pd.DataFrame({f'chain_{i+1}': all_n1[i] for i in range(N_samples)})
    df_n2 = pd.DataFrame({f'chain_{i+1}': all_n2[i] for i in range(N_samples)})

    # 保存每条链的 r 网格（独立保存为一个文件，或保存为多列 DataFrame）
    # 由于每条链的 r_max 不同，网格值不同，这里分别保存为单独的 CSV 文件（可选）
    # 简便起见，保存为一个字典格式的 CSV，每列是一条链的 r_vals
    # 注意：所有链的网格点数相同，但数值不同，可以按列存储
    df_r_vals = pd.DataFrame({f'chain_{i+1}': all_r_vals[i] for i in range(N_samples)})

    r1_path = os.path.join(output_dir, 'r1_values.csv')
    r2_path = os.path.join(output_dir, 'r2_values.csv')
    n1_path = os.path.join(output_dir, 'n1_values.csv')
    n2_path = os.path.join(output_dir, 'n2_values.csv')
    r_vals_path = os.path.join(output_dir, 'r_vals.csv')

    df_r1.to_csv(r1_path, index=False)
    df_r2.to_csv(r2_path, index=False)
    df_n1.to_csv(n1_path, index=False)
    df_n2.to_csv(n2_path, index=False)
    df_r_vals.to_csv(r_vals_path, index=False)

    print(f"\n所有结果已保存至目录：{output_dir}")
    print(f"  - xi_f.csv        : 采样的 xi_f 值")
    print(f"  - DeltaE.csv      : 计算得到的 ΔE 值")
    print(f"  - r_vals.csv      : 每条链的拉伸长度网格（每列一条链）")
    print(f"  - r1_values.csv   : 每条链的 r1 (每列一条链，共 {r_grids} 行)")
    print(f"  - r2_values.csv   : 每条链的 r2")
    print(f"  - n1_values.csv   : 每条链的 n1")
    print(f"  - n2_values.csv   : 每条链的 n2")

if __name__ == "__main__":
    main()
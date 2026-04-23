"""
目的：探究不同参数下的domain的解折叠顺序，参数包括xi_f1,xi_f2,E1,E2
系统：包含2个domain
主题：约束下优化系统的Helmholtz自由能
并行化：使用 multiprocessing.Pool 并行处理 (gamma, beta) 参数组合
"""

import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize_scalar
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============ 固定物理参数 ============
xi_f1 = 5.0      # 第一个domain的折叠态长度
k = 7.0          # k = xi_ui/xi_fi
E0 = 1.0

# ============ 优化参数 ============
r_grids = 1000

# ============ 存储路径 ============
save_path = "/home/tyt/project/Single-chain/opt+R/Rand_xi/Helmholtz_Optimization_results/2-domain_results"
os.makedirs(save_path, exist_ok=True)

# ============ 辅助函数（无全局变量依赖） ============
def energy_term_U(n_i, DeltaEi):
    """能量项: U(n_i) = ΔE_i n_i - ΔE_i cos(2π n_i)"""
    return DeltaEi * n_i - DeltaEi * np.cos(2 * np.pi * n_i)

def contour_length_Lci(n_i, xi_fi):
    """轮廓长度: L_{ci}(n_i) = ξ_fi + n_i (k - 1)ξ_fi"""
    return xi_fi + n_i * (k - 1) * xi_fi

def WLC_free_energy(x_i, L_ci):
    if x_i >= 1.0 - 1e-12:
        return 1e300
    return 0.25 * L_ci * (x_i**2 * (3.0 - 2.0 * x_i) / (1.0 - x_i))

def single_domain_free_energy(r_i, n_i, DeltaEi, xi_fi):
    L_ci = contour_length_Lci(n_i, xi_fi)
    if r_i < 0 or r_i > L_ci:
        return 1e300
    x_i = r_i / L_ci
    F_wlc = WLC_free_energy(x_i, L_ci)
    Ui = energy_term_U(n_i, DeltaEi)
    return F_wlc + Ui

def free_energy_2_domain(r, r1, n1, n2, gamma, beta):
    """两个域的自由能，参数 gamma, beta 作为输入"""
    # 第一个domain
    xi_f2 = beta * xi_f1
    delta_E1 = E0*(1 + gamma * (1 - beta))
    energy1 = single_domain_free_energy(r1, n1, delta_E1, xi_f1)
    # 第二个domain
    r2 = r - r1
    delta_E2 = E0
    energy2 = single_domain_free_energy(r2, n2, delta_E2, xi_f2)
    total = energy1 + energy2
    if not np.isfinite(total):
        return 1e300
    return total

def Optimize_single_point(r, gamma, beta, init_points=15, refine_levels=6, refine_points=15, tol=1e-6):
    """
    对给定拉伸 r 优化自由能，返回 [r, r1_opt, n1_opt, n2_opt]
    gamma, beta 作为参数传递，避免全局变量
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
                Lc2 = contour_length_Lci(n2, beta * xi_f1)
                r1_lower = max(0.0, r - min(r, Lc2))
                r1_upper = min(Lc1, r)
                if r1_lower > r1_upper:
                    continue

                # 对 r1 进行一维优化，传递 gamma, beta
                res = minimize_scalar(
                    lambda r1: free_energy_2_domain(r, r1, n1, n2, gamma, beta),
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

        idx_n1 = np.argmin(np.abs(n1_grid - best_n1))
        idx_n2 = np.argmin(np.abs(n2_grid - best_n2))
        left_n1 = n1_grid[max(0, idx_n1-1)]
        right_n1 = n1_grid[min(N-1, idx_n1+1)]
        left_n2 = n2_grid[max(0, idx_n2-1)]
        right_n2 = n2_grid[min(N-1, idx_n2+1)]
        n1_min, n1_max = left_n1, right_n1
        n2_min, n2_max = left_n2, right_n2

        if (n1_max - n1_min < tol) and (n2_max - n2_min < tol):
            break

    return np.array([r, best_r1, best_n1, best_n2])

# ============ 单个参数组合的计算任务 ============
def process_combination(params):
    """
    处理一个 (gamma, beta) 组合
    参数: params = (gamma_val, beta_val)
    返回: (gamma_val, beta_val, 状态字符串)
    """
    gamma_val, beta_val = params
    try:
        # 重新计算当前参数下的最大拉伸
        r_max = contour_length_Lci(1, xi_f1) + contour_length_Lci(1, beta_val * xi_f1)
        r_vals = np.linspace(0, 0.95 * r_max, r_grids)

        results = []
        for r in r_vals:
            opt_result = Optimize_single_point(r, gamma_val, beta_val)
            r_val, r1_opt, n1_opt, n2_opt = opt_result
            r2_opt = r_val - r1_opt if not np.isnan(r1_opt) else np.nan
            results.append([r_val, r1_opt, r2_opt, n1_opt, n2_opt])

        # 保存为CSV
        df = pd.DataFrame(results, columns=['r', 'r1', 'r2', 'n1', 'n2'])
        filename = f"gamma_{gamma_val:.1f}_beta_{beta_val:.1f}.csv"
        csv_filename = os.path.join(save_path, filename)
        df.to_csv(csv_filename, index=False)

        return (gamma_val, beta_val, f"成功保存至 {csv_filename}")
    except Exception as e:
        return (gamma_val, beta_val, f"失败: {str(e)}")

# ============ 主程序 ============
def main():
    # 参数扫描范围
    gamma_values = np.arange(1, 15.1, 1.0)   # 0.1, 0.2, 0.3, 0.4,0.5
    beta_values  = np.arange(0.1, 1.0, 0.2)   # 0.1, 0.3, 0.5, 0.7, 0.9

    # 生成所有参数组合
    param_combinations = [(a, b) for a in gamma_values for b in beta_values]
    total_combinations = len(param_combinations)

    print(f"共有 {total_combinations} 个参数组合待处理")
    print(f"使用 {os.cpu_count()} 个进程并行计算...")

    # 使用进程池并行处理
    with Pool(processes=os.cpu_count()) as pool:
        # 使用 imap_unordered 可以更快获取结果，但顺序不确定
        for i, result in enumerate(pool.imap_unordered(process_combination, param_combinations)):
            gamma_val, beta_val, msg = result
            print(f"[{i+1}/{total_combinations}] gamma={gamma_val:.1f}, beta={beta_val:.1f}: {msg}")

    print("\n所有参数组合处理完毕！")

if __name__ == "__main__":
    # 注意：在 Windows 上需要将 multiprocessing 代码放在 if __name__ 块内
    # 这里已经满足条件
    main()
import numpy as np
import matplotlib.pyplot as plt

# ========== 参数设置 ==========
# 设置l_p = k_B T = 1.0 作为单位量纲
# 所有能量和长度都以这些单位进行无量纲化

# 其他参数
N = 10  # 采样点数，也是n的最大值
xi_f = 2.0  # 折叠态片段长度（无量纲）
xi_u_mean = 30.0  # 未折叠态片段长度均值（无量纲）
xi_u_std = 10.0  # 未折叠态片段长度标准差（无量纲）

# 能量项参数（无量纲）
DeltaE = 3.0  # 能量差系数
U0 = 10.0  # 周期势系数

# 从正态分布采样 ξ_ui
np.random.seed(42)
xi_ui = np.random.normal(xi_u_mean, xi_u_std, int(N))
L = sum(xi_ui)
print(xi_ui)


# ========== 函数定义 ==========
def L_c_linear_interp(n, xi_ui):
    """
    使用线性插值计算Lc(n)
    根据图1 Step 1: Lc(n) = sum_{i=1}^n (ξ_ui - ξ_f)
    
    参数:
    n: 连续变量，0 ≤ n ≤ N
    xi_ui: 采样的未折叠片段长度数组
    xi_f: 折叠片段长度
    
    返回:
    轮廓长度Lc
    """
    Lc = np.zeros(N + 1)
    n_points = np.linspace(0, N, N+1)
    for i in range(N + 1):
        Lc[i] = L - sum(xi_ui[i:]) + (N - i)*xi_f

    Lc_fun = np.interp(n, n_points, Lc)
    
    return Lc_fun

def x_factor(r, n, xi_ui, xi_f):
    """
    无量纲端到端距离因子
    x(r,n) = r / L_c(n)
    """
    Lc = L_c_linear_interp(n, xi_ui)
    if Lc <= 0:
        return 0
    return r / Lc

def F_WLC(x, Lc):
    """
    无量纲Worm-like Chain (WLC) 自由能
    F_{WLC}(x, L_c) = (π^2)/(2L_c) * (1 - x^2) + (2L_c)/(π(1 - x^2))
    """
    if abs(x) >= 1.0:
        return np.inf
    
    term1 = (np.pi**2) / (2 * Lc) * (1 - x**2)
    term2 = (2 * Lc) / (np.pi * (1 - x**2))
    return term1 + term2

def U(n):
    """
    无量纲能量项
    U(n) = ΔE_n - U_0 * cos(2πn)
    """
    return DeltaE * n - U0 * np.cos(2 * np.pi * n)

def F_chain(r, n, xi_ui, xi_f):
    """
    无量纲单链自由能
    F_c(r,n) = F_{WLC}(r,n) + U(n)
    """
    # 确保n在有效范围内
    if n < 0 or n > N:
        return np.inf
    
    Lc = L_c_linear_interp(n, xi_ui)
    if Lc <= 0:
        return np.inf
    
    x = x_factor(r, n, xi_ui, xi_f)
    if abs(x) >= 1.0:
        return np.inf
    
    return F_WLC(x, Lc) + U(n)

def f_WLC(x, Lc):
    """
    无量纲WLC模型的力（从自由能导数得到）
    f_{WLC}(x, L_c) = -π^2 x / L_c^2 + 4x / (π(1 - x^2))
    """
    if abs(x) >= 1.0:
        return np.inf
    
    term1 = -np.pi**2 * x / Lc**2
    term2 = 4 * x / (np.pi * (1 - x**2))
    return term1 + term2

# ========== 均匀扫描优化 ==========
def optimize_n_for_r_uniform(r, xi_ui, xi_f, n_points=1000):
    """
    对于给定的r，使用均匀扫描找到最优的连续n（最小化自由能）
    """
    # 在[0, N]区间内均匀采样
    n_grid = np.linspace(0, N, n_points)
    F_grid = np.full(n_points, np.inf)
    
    # 计算每个n点的自由能
    for i, n_val in enumerate(n_grid):
        try:
            F_val = F_chain(r, n_val, xi_ui, xi_f)
            F_grid[i] = F_val
        except:
            F_grid[i] = np.inf
    
    # 找到最小自由能对应的n值
    min_idx = np.nanargmin(F_grid)
    n_opt = n_grid[min_idx]
    F_opt = F_grid[min_idx]
    
    return n_opt, F_opt

def calculate_f_r_uniform(r_range, xi_ui, xi_f, n_points=1000):
    """
    使用均匀扫描计算f(r)曲线
    """
    n_opt_list = []
    f_list = []
    
    total_points = len(r_range)
    
    for i, r in enumerate(r_range):
        # 找到最优n（使用均匀扫描）
        n_opt, F_opt = optimize_n_for_r_uniform(r, xi_ui, xi_f, n_points)
        n_opt_list.append(n_opt)
        
        # 计算对应的Lc和x
        Lc_opt = L_c_linear_interp(n_opt, xi_ui)
        x_opt = x_factor(r, n_opt, xi_ui, xi_f)
        
        # 计算力f
        if abs(x_opt) < 1.0 and Lc_opt > 0:
            f_val = f_WLC(x_opt, Lc_opt)
        else:
            f_val = np.inf
        
        f_list.append(f_val)
        
        # 显示进度
        if (i + 1) % max(1, total_points // 10) == 0:
            print(f"Progress: {i+1}/{total_points} ({(i+1)/total_points*100:.1f}%)")
    
    return np.array(n_opt_list), np.array(f_list)

# ========== 主计算 ==========
def main():
    print("Polymer Chain Free Energy Calculation (Using Linear Interpolation for Lc)")
    print("=" * 60)
    print("Note: All quantities are dimensionless")
    print("  - Length in units of persistence length l_p")
    print("  - Energy in units of k_B T")
    print("  - Force in units of k_B T/l_p")
    print("=" * 60)
    print(f"\nParameters: N={N}, L={L}, ξ_f={xi_f}")
    print(f"Sampled ξ_ui: {xi_ui}")
    print(f"ξ_ui mean: {np.mean(xi_ui):.2f}, standard deviation: {np.std(xi_ui):.2f}")
    
    # 创建r的范围（从0到最大可能的Lc）
    # 计算最大和最小Lc
    Lc_max = L_c_linear_interp(N, xi_ui)  # n=N时的Lc
    
    r_range = np.linspace(0, 0.95 * Lc_max, 1000)  # 避免x接近1
    
    # 计算最优n和力f
    print("\nStarting calculation of f(r) curve...")
    n_points = 500  # 每个r值对应的n均匀采样点数
    n_opt, f_values = calculate_f_r_uniform(r_range, xi_ui, xi_f, n_points)
    
    # 移除无穷大的值
    valid_indices = np.isfinite(f_values)
    r_valid = r_range[valid_indices]
    f_valid = f_values[valid_indices]
    n_valid = n_opt[valid_indices]
    
    # ========== 可视化 ==========
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. 采样ξ_ui的分布
    axes[0, 0].hist(xi_ui, bins=10, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    axes[0, 0].axvline(xi_u_mean, color='red', linestyle='--', label=f'Mean={xi_u_mean:.1f}')
    axes[0, 0].axvline(xi_f, color='green', linestyle='--', label=f'ξ_f={xi_f:.1f}')
    axes[0, 0].set_xlabel('ξ_ui (dimensionless)')
    axes[0, 0].set_ylabel('Probability Density')
    axes[0, 0].set_title('Distribution of Sampled ξ_ui')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. L_c(n)函数（通过线性插值）
    n_values = np.linspace(0, N, 1000)
    Lc_values = [L_c_linear_interp(n, xi_ui) for n in n_values]
    axes[0, 1].plot(n_values, Lc_values, 'b-', linewidth=1)
    axes[0, 1].set_xlabel('n (dimensionless)')
    axes[0, 1].set_ylabel('L_c(n) (dimensionless)')
    axes[0, 1].set_title('Contour Length L_c(n) (Linear Interpolation)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 能量项U(n)
    axes[0, 2].plot(n_values, U(n_values), 'g-', linewidth=2)
    axes[0, 2].set_xlabel('n (dimensionless)')
    axes[0, 2].set_ylabel('U(n) (dimensionless)')
    axes[0, 2].set_title('Energy Term U(n)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 最优n随r的变化
    axes[1, 0].plot(r_valid, n_valid, 'b.', markersize=3, alpha=0.7, label='Uniform scan results')
    axes[1, 0].set_xlabel('r (dimensionless)')
    axes[1, 0].set_ylabel('Optimal n (dimensionless)')
    axes[1, 0].set_title('Optimal Number of Folded Segments n(r)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 力f(r)曲线
    axes[1, 1].plot(r_valid, f_valid, 'r.', markersize=3, alpha=0.7, label='Uniform scan results')
    axes[1, 1].set_xlabel('r (dimensionless)')
    axes[1, 1].set_ylabel('f(r) (dimensionless)')
    axes[1, 1].set_title('Force-Extension Curve f(r)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 自由能景观示例（固定r）
    r_example = 50.0
    n_grid_example = np.linspace(0, N, 500)
    F_example = [F_chain(r_example, n, xi_ui, xi_f) for n in n_grid_example]
    n_min_idx = np.nanargmin(F_example)
    n_min = n_grid_example[n_min_idx]
    F_min = F_example[n_min_idx]
    
    axes[1, 2].plot(n_grid_example, F_example, 'm-', linewidth=1.5)
    axes[1, 2].plot(n_min, F_min, 'ro', markersize=6, label=f'Minimum: n={n_min:.2f}, F={F_min:.2f}')
    axes[1, 2].set_xlabel('n (dimensionless)')
    axes[1, 2].set_ylabel('F(r,n) (dimensionless)')
    axes[1, 2].set_title(f'Free Energy Landscape (r={r_example})')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    savepath = '/home/tyt/project/Single-chain/opt+R/Rand_xi/random_xi_partial_domain_results.png'
    plt.savefig(savepath, dpi=300)
    
    # ========== 输出结果 ==========
    print("\nCalculation Results Summary:")
    print("=" * 50)
    print(f"r calculation range: {r_range[0]:.2f} to {r_range[-1]:.2f} (dimensionless)")
    print(f"Optimal n range: {np.min(n_valid):.2f} to {np.max(n_valid):.2f} (dimensionless)")
    print(f"Force range: {np.min(f_valid):.4f} to {np.max(f_valid):.4f} (dimensionless)")
    print(f"Average force: {np.mean(f_valid):.4f} (dimensionless)")
    print(f"Force standard deviation: {np.std(f_valid):.4f} (dimensionless)")
    
    
    # 找出力最大和最小的点
    max_force_idx = np.argmax(f_valid)
    min_force_idx = np.argmin(f_valid)
    print(f"Maximum force: f({r_valid[max_force_idx]:.2f}) = {f_valid[max_force_idx]:.4f}, n = {n_valid[max_force_idx]:.2f}")
    print(f"Minimum force: f({r_valid[min_force_idx]:.2f}) = {f_valid[min_force_idx]:.4f}, n = {n_valid[min_force_idx]:.2f}")
    

    


if __name__ == "__main__":
    main()
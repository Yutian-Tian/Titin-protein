import numpy as np
from scipy.optimize import minimize

# 系统参数
L = 200.0    # 总长度
N = 5      
a = 5.0      
xi = 30.0     
U0 = 10.0
DeltaE = 10.0

l_p = 1.0
k_BT = 1.0


# 扫描参数
Rsteps = 1000  # R的采样点数

def F_WLC(R, r, n):
    """
    Worm-Like Chain 自由能作为R, r, n的函数
    """
    # 计算L_n = L - (N-n)ξ
    L_n = L - (N - n) * xi
    
    if L_n < 1e-12:
        return 0.0
    
    # 计算x_n = (R - r) / L_n
    x_n = (R - r) / L_n
    
    if np.abs(1 - x_n**2) < 1e-12:
        return 0.0
    
    term1 = (np.pi**2 * l_p) / (2.0 * L_n) * (1.0 - x_n**2)
    term2 = (2.0 * L_n) / (np.pi * l_p * (1.0 - x_n**2))
    return term1 + term2  # 因为k_B T = 1

def f_WLC(R, r, n):
    """
    WLC的力作为R, r, n的函数
    """
    # 计算L_n = L - (N-n)ξ
    L_n = L - (N - n) * xi
    
    if L_n < 1e-12:
        return 0.0
    
    # 计算x_n = (R - r) / L_n
    x_n = (R - r) / L_n
    
    if np.abs(1 - x_n**2) < 1e-12:
        return 0.0
    
    # WLC力的表达式
    # 根据F_WLC对r求导得到力
    term1 = (np.pi**2 * l_p) / L_n * x_n
    term2 = (4.0 * L_n * x_n) / (np.pi * l_p * (1.0 - x_n**2)**2)
    return term1 + term2


def F_FJC_from_WLC_force(R, r, n, num_points=50):
    """
    FJC自由能 - 通过对WLC的力从0到r积分得到
    F_FJC = ∫₀ʳ f_WLC(R, r', n) dr'
    """
    if r <= 0:
        return 0.0
    
    # 数值积分
    r_points = np.linspace(0, r, num_points)
    integrand = [f_WLC(R, r_i, n) for r_i in r_points]
    return np.trapezoid(integrand, r_points)

def U(n):
    if n < 0 or n > N:
        return np.inf
    return DeltaE * n - U0 * np.cos(2 * np.pi * n / N)

def F_total(R, r, n):
    """
    总自由能函数 F(R, r, n) = F_WLC(R, r, n) + ΔE_n(n) + F_FJC(R, r, n)
    """
    # 检查约束条件
    if not (0 <= r < (N - n) * a and 0 <= n <= N and r <= R < L):
        return np.inf  # 不满足约束，返回无穷大
    
    F_wlc = F_WLC(R, r, n)
    F_fjc = F_FJC_from_WLC_force(R, r, n)
    
    return F_wlc + U(n) + F_fjc

def check_constraints(R, r, n):
    """
    检查约束条件是否满足
    """
    constraints = [
        (0 <= r, "r >= 0"),
        (r < (N - n) * a, f"r < {(N - n) * a}"),
        (0 <= n, "n >= 0"),
        (n <= N, f"n <= {N}"),
        (r <= R, "r <= R"),
        (R < L, f"R < {L}")
    ]
    
    violations = []
    for condition, message in constraints:
        if not condition:
            violations.append(message)
    
    return len(violations) == 0, violations

def find_optimal_r_n(R):
    """
    对于给定的R，找到使F(R, r, n)最小的(r, n)
    """
    def objective(params):
        r, n = params
        return F_total(R, r, n)
    
    # 设置边界条件
    r_max = min(R, (N-0.01)*a)  # 稍微小于理论最大值
    bounds = [(0.0, r_max), (0.0, N-0.01)]
    
    # 多个初始猜测点
    initial_guesses = [
        [min(R/2, (N/2)*a/2), N/2],      # 中点
        [min(R/4, (N/4)*a/2), N/4],      # 小r小n
        [min(3*R/4, (3*N/4)*a/2), 3*N/4], # 大r大n
        [min(R/10, a), 10],               # 很小r
        [min(9*R/10, (N-10)*a), N-10]     # 很大r
    ]
    
    best_result = None
    min_F = np.inf
    
    for guess in initial_guesses:
        try:
            result = minimize(objective, guess, bounds=bounds, method='L-BFGS-B', 
                            options={'ftol': 1e-8, 'maxiter': 1000})
            
            if result.success and result.fun < min_F:
                min_F = result.fun
                best_result = result
        except:
            continue
    
    if best_result is None:
        # 如果优化失败，使用网格搜索
        print(f"Warning: Optimization failed for R={R}, using grid search")
        return find_optimal_grid(R)
    
    r_opt, n_opt = best_result.x
    
    # 检查约束条件
    valid, violations = check_constraints(R, r_opt, n_opt)
    if not valid:
        print(f"Warning: Constraints violated for R={R}: {violations}")
    
    return r_opt, n_opt, min_F

def find_optimal_grid(R):
    """
    网格搜索备选方案
    """
    # 粗网格搜索
    n_points = min(50, N)
    r_points = min(50, int(R * 10))
    
    n_vals = np.linspace(0, N, n_points)
    r_vals = np.linspace(0, min(R, (N-0.1)*a), r_points)
    
    min_F = np.inf
    optimal_r = 0.0
    optimal_n = 0.0
    
    for n_val in n_vals:
        for r_val in r_vals:
            if r_val < (N - n_val) * a and r_val <= R:
                F_val = F_total(R, r_val, n_val)
                if F_val < min_F:
                    min_F = F_val
                    optimal_r = r_val
                    optimal_n = n_val
    
    return optimal_r, optimal_n, min_F

def analyze_energy_contributions(R, r, n):
    """
    分析各能量部分的贡献
    """
    F_wlc = F_WLC(R, r, n)
    F_fjc = F_FJC_from_WLC_force(R, r, n)
    total_F = F_wlc + F_fjc + U(n)
    
    print(f"Energy Analysis for R={R}, r={r}, n={n}:")
    print(f"  F_WLC = {F_wlc:.6f}")
    print(f"  F_FJC = {F_fjc:.6f}")
    print(f"  ΔE_n = {U(n):.6f}")
    print(f"  Total F = {total_F:.6f}")
    
    # 计算L_n和x_n用于分析
    L_n = L - (N - n) * xi
    if L_n > 1e-12:
        x_n = (R - r) / L_n
        print(f"  L_n = {L_n:.4f}, x_n = {x_n:.4f}")
    
    return F_wlc, F_fjc, U(n)

def main():
    """
    主函数：扫描R值，找到最优的(r, n)
    """
    R_values = np.linspace(0.1, L * 0.999, Rsteps)  # 避免L_n=0的情况
    
    results = []
    energy_contributions = []
    
    print("Scanning R values with l_p = k_B T = 1, continuous n...")
    print("R\t\tr_opt\t\tn_opt\t\tF_min\t\tF_WLC\t\tF_FJC")
    print("-" * 70)
    
    for i, R_val in enumerate(R_values):
        r_opt, n_opt, F_min = find_optimal_r_n(R_val)
        
        # 计算各部分的贡献
        F_wlc = F_WLC(R_val, r_opt, n_opt)
        F_fjc = F_FJC_from_WLC_force(R_val, r_opt, n_opt)
        
        results.append((R_val, r_opt, n_opt, F_min, F_wlc, F_fjc))
        
        if i % 100 == 0 or i == Rsteps - 1:
            print(f"{R_val:.4f}\t{r_opt:.4f}\t{n_opt:.4f}\t{F_min:.6f}\t{F_wlc:.6f}\t{F_fjc:.6f}")
        
        # 对于某些点进行详细分析
        if i % 20 == 0:
            energy_contributions.append(analyze_energy_contributions(R_val, r_opt, n_opt))
    
    return np.array(results)

# 运行主程序
if __name__ == "__main__":
    results = main()
    
    # 保存结果
    np.savetxt('WLC+Langvin_optimization_results_no_inverse.csv', 
               results, 
               delimiter=',', 
               header='R,r_opt,n_opt,F_min,F_WLC,F_FJC',
               comments='',
               fmt='%.6f')
    
    print("\nOptimization completed. Results saved to WLC+Langvin_optimization_results_no_inverse.csv")
    
    # 简单绘图
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        
        # 1. 最优参数 vs R
        plt.subplot(2, 3, 1)
        plt.plot(results[:, 0], results[:, 1], 'b-', label='Optimal r')
        plt.xlabel('R')
        plt.ylabel('Optimal r')
        plt.title('Optimal r vs R')
        plt.legend()
        
        plt.subplot(2, 3, 2)
        plt.plot(results[:, 0], results[:, 2], 'r-', label='Optimal n')
        plt.xlabel('R')
        plt.ylabel('Optimal n')
        plt.title('Optimal n vs R')
        plt.legend()
        
        # 2. 自由能分解
        plt.subplot(2, 3, 3)
        plt.plot(results[:, 0], results[:, 3], 'k-', label='Total F')
        plt.plot(results[:, 0], results[:, 4], 'g--', label='F_WLC')
        plt.plot(results[:, 0], results[:, 5], 'm--', label='F_FJC')
        plt.xlabel('R')
        plt.ylabel('Free Energy')
        plt.title('Free Energy Components')
        plt.legend()
        
        # 3. 归一化参数
        plt.subplot(2, 3, 4)
        plt.plot(results[:, 0], results[:, 1]/results[:, 0], 'b-', label='r/R')
        plt.plot(results[:, 0], results[:, 2]/N, 'r-', label='n/N')
        plt.xlabel('R')
        plt.ylabel('Normalized values')
        plt.title('Normalized Optimal Parameters')
        plt.legend()
        
        # 4. 力平衡检查
        plt.subplot(2, 3, 5)
        # 计算WLC的力
        f_wlc_values = [f_WLC(R, r, n) for R, r, n in zip(results[:, 0], results[:, 1], results[:, 2])]
        plt.plot(results[:, 0], f_wlc_values, 'g-', label='f_WLC')
        plt.xlabel('R')
        plt.ylabel('Force')
        plt.title('WLC Force vs R')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('WLC+Langvin_optimization_results_no_inverse.png', dpi=300)
        
    except ImportError:
        print("Matplotlib not available for plotting")
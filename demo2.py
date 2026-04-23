import numpy as np
from scipy.optimize import minimize

# 系统参数
L = 200.0    # 总长度
N = 5      
a = 5.0      
xi = 30.0     
U0 = 10.0
DeltaE = 10.0

# 扫描参数
Rsteps = 100

def F_WLC(x_n, L_n):
    """
    Worm-Like Chain 自由能
    """
    if L_n < 1e-12:
        return 0.0
    
    # 检查x_n范围
    if x_n <= -1.0 or x_n >= 1.0:
        return np.inf
    
    if abs(x_n) < 1e-6:
        return 0.0
    
    term = 1.0 - x_n**2
    if term < 1e-12:
        return np.inf
        
    return (2.0 * L_n) / (np.pi * term)

def U(n):
    """ΔE_n项"""
    # 检查n范围
    if n < 0 or n > N:
        return np.inf
    return DeltaE * n - U0 * np.cos(2 * np.pi * n / N)

def F_total(R, r, n):
    """
    总自由能函数
    检查所有约束条件
    """
    # 检查约束条件1: 0 <= n <= N
    if n < 0 or n > N:
        return np.inf
    
    # 检查约束条件2: 0 <= r < (N - n) * a
    if r < 0 or r >= (N - n) * a:
        return np.inf
    
    # 检查约束条件3: r <= R < L
    if R < r or R >= L:
        return np.inf
    
    # 计算L_n
    L_n = L - (N - n) * xi
    if L_n < 0:
        return np.inf
    
    # 计算x_n
    if L_n < 1e-12:
        x_n = 0.0
    else:
        x_n = (R - r) / L_n
        # 检查x_n是否在有效范围内
        if x_n <= -1.0 or x_n >= 1.0:
            return np.inf
    
    F_wlc = F_WLC(x_n, L_n)
    U_val = U(n)
    
    return F_wlc + U_val

def find_optimal_r_n(R):
    """
    对于给定的R，找到使F(R, r, n)最小的(r, n)
    确保在约束范围内搜索
    """
    if R < 0 or R >= L:
        return 0.0, 0.0, np.inf
    
    def objective(params):
        r, n = params
        return F_total(R, r, n)
    
    # 设置合理的边界，确保在约束范围内
    # r的范围: 0 <= r < min(R, (N-0.1)*a)
    r_max = min(R, (N-0.1)*a)
    bounds = [(0.0, r_max), (0.0, N)]
    
    # 尝试多个初始点，确保在可行域内
    initial_guesses = []
    
    # 生成一些在约束范围内的初始猜测
    for n_guess in [0.1, N/4, N/2, 3*N/4, N-0.1]:
        if n_guess < 0 or n_guess > N:
            continue
        
        max_r_for_n = min(R, (N - n_guess) * a * 0.99)
        if max_r_for_n > 0:
            for r_frac in [0.2, 0.5, 0.8]:
                r_guess = r_frac * max_r_for_n
                if r_guess >= 0 and r_guess < (N - n_guess) * a:
                    initial_guesses.append([r_guess, n_guess])
    
    best_F = np.inf
    best_r = 0.0
    best_n = 0.0
    
    for guess in initial_guesses:
        try:
            result = minimize(objective, guess, bounds=bounds, 
                             method='L-BFGS-B', options={'ftol': 1e-6, 'maxiter': 100})
            
            # 检查结果是否满足约束条件
            r_opt, n_opt = result.x
            if (result.success and 
                result.fun < best_F and
                n_opt >= 0 and n_opt <= N and
                r_opt >= 0 and r_opt < (N - n_opt) * a and
                R >= r_opt and R < L):
                best_F = result.fun
                best_r, best_n = result.x
        except:
            continue
    
    return best_r, best_n, best_F

def main():
    """主函数：扫描R值，找到最优的(r, n)"""
    # R的范围: 0 <= R < L
    R_values = np.linspace(1.0, L-1.0, Rsteps)
    results = []
    
    print("Scanning R values with constraint checking...")
    print("R\t\tr_opt\t\tn_opt\t\tF_min")
    print("-" * 50)
    
    for i, R_val in enumerate(R_values):
        r_opt, n_opt, F_min = find_optimal_r_n(R_val)
        results.append((R_val, r_opt, n_opt, F_min))
        
        if i % 10 == 0:
            print(f"{R_val:.2f}\t{r_opt:.4f}\t{n_opt:.4f}\t{F_min:.6f}")
    
    return np.array(results)

if __name__ == "__main__":
    results = main()
    
    # 保存结果
    np.savetxt('optimization_results_no_inverse.csv', results, delimiter=',', 
               header='R,r_opt,n_opt,F_min', comments='', fmt='%.6f')
    
    print("\nOptimization completed. Results saved to optimization_results_no_inverse.csv")
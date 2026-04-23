import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 参数设置
L = 200.0
xi = 30.0
a = 5.0
N = 5
deltaE = 10.0
U0 = 10.0

global Rsteps
global nsteps
global rsteps

Rsteps = 1000
nsteps = 1000
rsteps = 1000

# 定义各种函数
def F_WLC(R, r, n_u):
    """计算WLC部分的自由能"""
    Ln_val = L - (N - n_u) * xi
    # 避免除零错误
    if Ln_val <= 0:
        return float('inf')
    
    xn_val = (R - r) / Ln_val
    
    # 避免1-x_n^2为负数或接近零
    if abs(xn_val) >= 1 or abs(1 - xn_val**2) < 1e-10:
        return float('inf')
    
    term1 = (np.pi**2) / (2 * Ln_val) * (1 - xn_val**2)  
    term2 = (2 * Ln_val) / (np.pi * (1 - xn_val**2))
    
    return term1 + term2

def F_FJC(r, n_u):
    """计算FJC部分的自由能"""
    if n_u >= N:  # 避免除零
        return float('inf')
    return (3 * r**2) / (2 * (N - n_u) * a**2 + 1e-10) + 1.5* np.log(N - n_u + 1e-10)

def total_F(R, r, n_u, delta_E=deltaE):
    """计算总自由能F"""
    return F_WLC(R, r, n_u) + delta_E * n_u - U0*np.cos(2 * np.pi * n_u) + F_FJC(r, n_u)

def dF_dR(R, r, n_u, h=1e-6):
    # 计算中间变量
    Lc = L - (N - n_u) * xi
    x_val = (R - r) / Lc
    
    # 计算导数
    d_term1_dr = - np.pi**2 * x_val / Lc**2
    d_term2_dr = 4 * x_val / (np.pi * (1 - x_val**2))
    
    # U(nf)不依赖于r，其导数为0
    return d_term1_dr + d_term2_dr

def check_constraints(R, r, n_u):
    """检查约束条件是否满足"""
    # 约束1: 0 <= r <= n_u * a
    if not (0 <= r <= n_u * a):
        return False
    
    # 约束2: 0 <= R - r <= L - (N - n_u) * xi
    Ln_val = L - (N - n_u) * xi
    if Ln_val <= 0:
        return False
    if not (0 <= (R - r) <= Ln_val):
        return False
    
    return True

# 主计算函数
def find_optimal_parameters(R_values):
    """对于每个R值，找到最优的n_u和r"""
    results = []
    
    for i, R in enumerate(R_values):
        min_F = float('inf')
        best_n_u = None
        best_r = None
        best_F = None
        
        # 扫描所有可能的n_u值（连续值）
        n_u_values = np.linspace(0, N-1e-10, nsteps)
        
        for n_u in n_u_values:
            # 对于每个n_u，r的范围是[0, min(n_u*a, R)]，但要考虑约束
            r_max = min(n_u * a, R)  # r <= n_u*a 且 r <= R (因为R-r>=0)
            if r_max <= 0:
                continue
                
            r_values = np.linspace(0, r_max, rsteps)
            
            for r in r_values:
                # 检查所有约束条件
                if not check_constraints(R, r, n_u):
                    continue
                
                # 计算当前参数下的F值
                current_F = total_F(R, r, n_u)
                
                # 更新最小值
                if current_F < min_F:
                    min_F = current_F
                    best_n_u = n_u
                    best_r = r
                    best_F = current_F
        
        # 如果没有找到满足约束的解，使用边界值
        if best_n_u is None:
            # 尝试找到边界解
            best_n_u = 0
            best_r = 0
            best_F = total_F(R, 0, 0)
        
        # 计算∂F/∂R
        dF_dR_val = dF_dR(R, best_r, best_n_u)
        
        results.append({
            'R': R,
            'n_u': best_n_u,
            'r': best_r,
            'F': best_F,
            'dF_dR': dF_dR_val
        })
        
        # 显示进度
        if (i + 1) % 100 == 0:
            print(f"处理进度: {i + 1}/{len(R_values)}, R={R:.2f}, r={best_r:.3f}, 最优nu={best_n_u:.6f}, 最小F={best_F:.6f}, ∂F/∂R={dF_dR_val:.6f}")
    
    return pd.DataFrame(results)

# 生成R值范围
R_values = np.linspace(0, L-0.01, Rsteps)  # [0, L)

# 进行计算
print("开始计算...")
results_df = find_optimal_parameters(R_values)
print("计算完成!")

# 保存结果
results_df.to_csv('WLC+Gauss_results.csv', index=False)
print("结果已保存到 WLC+Gauss_results.csv")

# 绘制图形
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# F-R 曲线
ax1.plot(results_df['R'], results_df['F'])
ax1.set_xlabel('R')
ax1.set_ylabel('F')
ax1.set_title('Free Energy vs R')
ax1.grid(True)

# n_u-R 曲线
ax2.plot(results_df['R'], results_df['n_u'], 'o-', markersize=2)
ax2.set_xlabel('R')
ax2.set_ylabel('n_u')
ax2.set_title('Optimal n_u vs R')
ax2.grid(True)

# ∂F/∂R-R 曲线
ax3.plot(results_df['R'], results_df['dF_dR'])
ax3.set_xlabel('R')
ax3.set_ylabel('∂F/∂R')
ax3.set_title('Force vs R')
ax3.grid(True)

# r-R 曲线
ax4.plot(results_df['R'], results_df['r'], 'o-', markersize=2)
ax4.set_xlabel('R')
ax4.set_ylabel('r')
ax4.set_title('Optimal r vs R')
ax4.grid(True)

plt.tight_layout()
plt.savefig('results_WLC+Gauss.png', dpi=300, bbox_inches='tight')

# 输出一些统计信息
print(f"\n统计信息:")
print(f"R范围: {results_df['R'].min():.2f} 到 {results_df['R'].max():.2f}")
print(f"F范围: {results_df['F'].min():.4f} 到 {results_df['F'].max():.4f}")
print(f"n_u范围: {results_df['n_u'].min():.4f} 到 {results_df['n_u'].max():.4f}")
print(f"r范围: {results_df['r'].min():.4f} 到 {results_df['r'].max():.4f}")

# 验证约束条件
print(f"\n约束条件验证:")
for i in [0, len(results_df)//4, len(results_df)//2, 3*len(results_df)//4, -1]:
    row = results_df.iloc[i]
    R, r, n_u = row['R'], row['r'], row['n_u']
    constraint1 = 0 <= r <= n_u * a
    constraint2 = 0 <= (R - r) <= (L - (N - n_u) * xi)
    print(f"R={R:.2f}: r={r:.2f} in [0, {n_u*a:.2f}]? {constraint1}, R-r={R-r:.2f} in [0, {L-(N-n_u)*xi:.2f}]? {constraint2}")
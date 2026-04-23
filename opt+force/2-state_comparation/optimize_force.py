import numpy as np
import pandas as pd
import os

# ==================== 参数设置 ====================
np.random.seed(42)  # 可重复性
num_samples = 100   # 采样100个ξ_f
E = 3.0            # ΔE = 3
U0 = 10.0          # U0 = 10.0

# ξ_f的分布参数
mu_xi_f = 3.0      # 均值
sigma_xi_f = 0.5   # 标准差

# 网格设置
n_grid_points = 100    # n的网格点数 (0到1)
r_grid_points = 1000   # r的网格点数

# f值范围
f_min = 0.0
f_max = 10.0
f_points = 1000     # f的采样点数

# 输出路径
output_dir = '/home/tyt/project/Single-chain/opt+force/2-state_comparation/results'

# ==================== 辅助函数 ====================
def L_c(n, xi_f, xi_u):
    """轮廓长度"""
    return xi_f + n * (xi_u - xi_f)

def x_value(r, n, xi_f, xi_u):
    """端到端因子"""
    L = L_c(n, xi_f, xi_u)
    if L <= 0:
        return np.nan
    return r / L

def F_WLC(x, L_c_val):
    """WLC自由能"""
    if x >= 1 or x <= -1 or L_c_val <= 0:
        return np.inf
    term1 = (np.pi**2) / (2 * L_c_val) * (1 - x**2)
    term2 = (2 * L_c_val) / (np.pi * (1 - x**2))
    return term1 + term2

def U(n):
    """能量项"""
    return E * n - U0 * np.cos(2 * np.pi * n)

def F_total(r, n, xi_f, xi_u, f):
    """总自由能: F_c(r, n; f) = F_WLC(r, n) + U(n) - f·r"""
    L = L_c(n, xi_f, xi_u)
    x = x_value(r, n, xi_f, xi_u)
    
    if np.isnan(x):
        return np.inf
        
    return F_WLC(x, L) + U(n) - f * r

def optimize_for_f(f_value, xi_f, xi_u):
    """对于给定的f，优化r和n"""
    # 创建网格
    n_values = np.linspace(0, 1, n_grid_points)
    r_values = np.linspace(0, xi_u, r_grid_points)
    
    # 初始化最小值
    min_energy = np.inf
    best_r = 0.0
    best_n = 0.0
    
    # 网格扫描优化
    for n in n_values:
        for r in r_values:
            # 检查r是否超过当前n对应的轮廓长度
            L = L_c(n, xi_f, xi_u)
            if r > L:
                continue
                
            energy = F_total(r, n, xi_f, xi_u, f_value)
            
            if energy < min_energy:
                min_energy = energy
                best_r = r
                best_n = n
    
    return best_r, best_n

def process_sample(sample_idx, xi_f, xi_u):
    """处理单个样本"""
    f_values = np.linspace(f_min, f_max, f_points)
    
    # 存储结果
    r_results = []
    n_results = []
    
    for i, f in enumerate(f_values):
        # 显示进度
        if i % 10 == 0:
            print(f"  采样{sample_idx+1}: 处理f={f:.2f} ({i+1}/{len(f_values)})")
        
        # 优化
        r_opt, n_opt = optimize_for_f(f, xi_f, xi_u)
        
        # 存储结果
        r_results.append(r_opt)
        n_results.append(n_opt)
    
    return r_results, n_results, f_values

# ==================== 主函数 ====================
def main():
    """主函数"""
    print("=" * 60)
    print("对比2-state理论与定力系综的结果")
    print(f"参数: ΔE={E}, U0={U0}")
    print(f"采样数: {num_samples}")
    print(f"ξ_f分布: N({mu_xi_f}, {sigma_xi_f}²)")
    print(f"ξ_u = 10 * ξ_f")
    print("=" * 60)
    
    # Step 1: 采样ξ_f
    print(f"\nStep 1: 采样{num_samples}个ξ_f...")
    xi_f_samples = np.random.normal(mu_xi_f, sigma_xi_f, num_samples)
    xi_u_samples = 10 * xi_f_samples
    
    # 显示统计信息
    print(f"ξ_f统计: 均值={np.mean(xi_f_samples):.3f}, 标准差={np.std(xi_f_samples):.3f}")
    print(f"ξ_f范围: [{np.min(xi_f_samples):.3f}, {np.max(xi_f_samples):.3f}]")
    print(f"ξ_u统计: 均值={np.mean(xi_u_samples):.3f}, 标准差={np.std(xi_u_samples):.3f}")
    print(f"ξ_u范围: [{np.min(xi_u_samples):.3f}, {np.max(xi_u_samples):.3f}]")
    
    # 存储所有采样的结果
    all_r = []
    all_n = []
    
    # Step 2: 对每个采样进行优化
    print(f"\nStep 2: 对每个采样进行优化...")
    for sample_idx in range(num_samples):
        print(f"处理采样 {sample_idx+1}/{num_samples}...")
        
        xi_f = xi_f_samples[sample_idx]
        xi_u = xi_u_samples[sample_idx]
        
        # 处理单个采样
        r_results, n_results, f_values = process_sample(sample_idx, xi_f, xi_u)
        
        # 添加到总结果
        all_r.append(r_results)
        all_n.append(n_results)
    
    # f_values对于所有采样都是相同的
    f_values = np.linspace(f_min, f_max, f_points)
    
    # Step 3: 保存结果
    print(f"\nStep 3: 保存结果到文件...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建DataFrame并保存
    df_r = pd.DataFrame(np.array(all_r).T)  # 转置，使每列对应一个采样
    df_n = pd.DataFrame(np.array(all_n).T)
    df_f = pd.DataFrame(f_values.reshape(-1, 1))  # f值只有一列
    
    # 保存为CSV文件
    r_file_path = os.path.join(output_dir, 'r_values.csv')
    n_file_path = os.path.join(output_dir, 'n_values.csv')
    f_file_path = os.path.join(output_dir, 'f_values.csv')
    
    df_r.to_csv(r_file_path, index=False)
    df_n.to_csv(n_file_path, index=False)
    df_f.to_csv(f_file_path, index=False)
    
    print(f"\n结果已保存:")
    print(f"  - r_values.csv: {df_r.shape} (行数={df_r.shape[0]}, 列数={df_r.shape[1]})")
    print(f"  - n_values.csv: {df_n.shape} (行数={df_n.shape[0]}, 列数={df_n.shape[1]})")
    print(f"  - f_values.csv: {df_f.shape} (行数={df_f.shape[0]}, 列数={df_f.shape[1]})")
    
    # 保存参数信息
    params_df = pd.DataFrame({
        'sample_id': range(1, num_samples + 1),
        'xi_f': xi_f_samples,
        'xi_u': xi_u_samples
    })
    params_file = os.path.join(output_dir, 'parameters.csv')
    params_df.to_csv(params_file, index=False)
    
    # 保存配置信息
    config_df = pd.DataFrame({
        'parameter': ['num_samples', 'E', 'U0', 'mu_xi_f', 'sigma_xi_f', 
                      'f_min', 'f_max', 'f_points', 'n_grid_points', 'r_grid_points'],
        'value': [num_samples, E, U0, mu_xi_f, sigma_xi_f, 
                  f_min, f_max, f_points, n_grid_points, r_grid_points],
        'description': [
            '采样数量',
            'ΔE值',
            'U0值',
            'ξ_f分布的均值',
            'ξ_f分布的标准差',
            'f的最小值',
            'f的最大值',
            'f的采样点数',
            'n的网格点数',
            'r的网格点数'
        ]
    })
    config_file = os.path.join(output_dir, 'config.csv')
    config_df.to_csv(config_file, index=False)
    
    print("\n" + "=" * 60)
    print("计算完成!")
    print(f"结果保存在: {output_dir}")
    print("=" * 60)
    
    # 显示一些统计信息
    print("\n统计信息:")
    print("-" * 40)
    
    # 计算平均曲线
    avg_r = np.mean(np.array(all_r), axis=0)
    avg_n = np.mean(np.array(all_n), axis=0)
    
    print(f"f值范围: [{f_min:.1f}, {f_max:.1f}]，共{f_points}个点")
    print(f"平均r值范围: [{np.min(avg_r):.3f}, {np.max(avg_r):.3f}]")
    print(f"平均n值范围: [{np.min(avg_n):.3f}, {np.max(avg_n):.3f}]")
    
    # 创建并保存平均曲线
    avg_df = pd.DataFrame({
        'f': f_values,
        'avg_r': avg_r,
        'avg_n': avg_n
    })
    avg_file = os.path.join(output_dir, 'average_curve.csv')
    avg_df.to_csv(avg_file, index=False)
    print(f"平均曲线已保存到: {avg_file}")

if __name__ == "__main__":
    main()
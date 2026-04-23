import numpy as np
import pandas as pd
import os

# 全局参数设置
np.random.seed(42)  # 为了可重复性
num_samples = 200
k = 10.0  # ξ_u = k * ξ_f
Ek = 0.01  # 能量缩放系数
Uk = 0.02  # 周期势缩放系数
n_grid_points = 10000  # n的均匀划分点数
r_grid_points = 1000  # r值点数
mu = 10.0 
sigma = 5.0

# 指定输出路径
output_dir = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results'

# WLC自由能函数 (k_B T和l_p设为1，不显式出现在公式中)
def F_WLC(x, L_c):
    """Worm-like chain自由能 (k_BT=1, l_p=1)"""
    # 避免除以零或接近零
    if x >= 1 or x <= -1 or L_c <= 0:
        return np.inf
    
    term1 = (np.pi**2) / (2 * L_c) * (1 - x**2)
    term2 = (2 * L_c) / (np.pi * (1 - x**2))
    return term1 + term2  # k_BT=1，所以省略

# U(n)函数
def U_n(n, xi_u):
    """额外的能量项"""
    delta_E = Ek * xi_u**3
    U0 = Uk*xi_u**3
    return delta_E * n - U0 * np.cos(2 * np.pi * n)

# 总自由能函数
def total_free_energy(n, r, xi_f, xi_u):
    """单个链的总自由能"""
    # 计算轮廓长度
    L_c = xi_f + n * (xi_u - xi_f)
    
    # 计算x
    x = r / L_c
    
    # 计算自由能
    F_wlc = F_WLC(x, L_c)
    U = U_n(n, xi_u)
    
    return F_wlc + U

# 力的计算函数 (k_B T和l_p设为1，不显式出现在公式中)
def calculate_force(r, n, xi_f, xi_u):
    """根据给定的n计算力f (k_BT=1, l_p=1)"""
    # 计算轮廓长度
    L_c = xi_f + n * (xi_u - xi_f)
    
    if L_c <= 0:
        return np.nan
    
    # 计算x
    x = r / L_c
    
    # 检查x是否在有效范围内
    if x >= 1 or x <= -1:
        return np.nan
    
    # 根据公式计算力 (k_BT=1, l_p=1，所以公式中没有这些因子)
    term1 = - (np.pi**2 * x) / (L_c**2)
    term2 = (4 * x) / (np.pi * (1 - x**2)**2)
    
    return term1 + term2

def optimize_for_sample(xi_f, xi_u, n_grid):
    """为单个采样执行优化"""
    # 为采样生成r值范围：从0到0.95*ξ_u
    r_max = 0.99 * xi_u
    r_values = np.linspace(0, r_max, r_grid_points)
    
    # 存储当前采样的结果
    current_n = []
    current_f = []
    
    # 对每个r值进行优化
    for r in r_values:
        # 初始化最小自由能和对应的n
        min_energy = np.inf
        best_n = 0.5  # 默认值
        
        # 均匀扫描n的可行域
        for n in n_grid:
            # 计算自由能
            energy = total_free_energy(n, r, xi_f, xi_u)
            
            # 更新最小值
            if energy < min_energy:
                min_energy = energy
                best_n = n
        
        # 使用找到的最佳n计算力
        n_opt = best_n
        f_opt = calculate_force(r, n_opt, xi_f, xi_u)
        
        current_n.append(n_opt)
        current_f.append(f_opt)
    
    return r_values, current_n, current_f

def save_results(all_r, all_n, all_f, xi_f_samples, xi_u_samples):
    """保存结果到文件（优化版，避免性能警告）"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 方法1：使用字典一次性创建DataFrame（避免性能警告）
    # 创建字典，键为列名，值为数据
    data_dict_r = {f'Sample_{i+1}': all_r[i] for i in range(len(all_r))}
    data_dict_n = {f'Sample_{i+1}': all_n[i] for i in range(len(all_n))}
    data_dict_f = {f'Sample_{i+1}': all_f[i] for i in range(len(all_f))}
    
    # 一次性创建DataFrame
    df_r = pd.DataFrame(data_dict_r)
    df_n = pd.DataFrame(data_dict_n)
    df_f = pd.DataFrame(data_dict_f)
    
    # 保存为CSV文件到指定路径
    r_file_path = os.path.join(output_dir, 'r_values.csv')
    n_file_path = os.path.join(output_dir, 'n_values.csv')
    f_file_path = os.path.join(output_dir, 'f_values.csv')
    
    df_r.to_csv(r_file_path, index=False)
    df_n.to_csv(n_file_path, index=False)
    df_f.to_csv(f_file_path, index=False)
    
    print(f"\n结果已保存到CSV文件:")
    print(f"- r_values.csv: {r_file_path} (形状: {df_r.shape})")
    print(f"- n_values.csv: {n_file_path} (形状: {df_n.shape})")
    print(f"- f_values.csv: {f_file_path} (形状: {df_f.shape})")
    
    # 保存采样参数和统计信息
    params_df = pd.DataFrame({
        'xi_f': xi_f_samples,
        'xi_u': xi_u_samples,
        'r_max': 0.95 * xi_u_samples  # 每个采样的最大r值
    })
    params_file_path = os.path.join(output_dir, 'parameters.csv')
    params_df.to_csv(params_file_path, index=False)
    
    print(f"\n参数信息已保存到: {params_file_path}")
    
    # 保存参数设置信息
    config_df = pd.DataFrame({
        'parameter': ['num_samples', 'k', 'Ek', 'Uk', 'n_grid_points', 'num_r_points'],
        'value': [num_samples, k, Ek, Uk, n_grid_points, r_grid_points],
        'description': [
            '采样数量',
            'ξ_u = k * ξ_f 的系数',
            '能量缩放系数 (delta_E = Ek * ξ_u^3)',
            '周期势缩放系数 (U0 = Uk * ξ_u^3)',
            'n的均匀划分点数',
            '每个采样的r值点数'
        ]
    })
    config_file_path = os.path.join(output_dir, 'config.csv')
    config_df.to_csv(config_file_path, index=False)
    print(f"配置信息已保存到: {config_file_path}")
    
    return r_file_path, n_file_path, f_file_path

def sample_xi_f_with_constraint(mu, sigma, num_samples, lower_bound, upper_bound):
    """采样ξ_f，只保留在[lower_bound, upper_bound]之间的样本"""
    xi_f_samples = []
    xi_u_samples = []
    
    print(f"正在采样ξ_f，要求范围: [{lower_bound}, {upper_bound}]")
    
    # 继续采样直到达到所需数量
    while len(xi_f_samples) < num_samples:
        # 每次采样一批，数量为所需数量的2倍，以提高效率
        batch_size = min(num_samples * 2, num_samples - len(xi_f_samples) + 100)
        batch_xi_f = np.random.normal(mu, sigma, batch_size)
        
        # 筛选符合条件的样本
        mask = (batch_xi_f >= lower_bound) & (batch_xi_f <= upper_bound)
        valid_xi_f = batch_xi_f[mask]
        valid_xi_u = k * valid_xi_f
        
        # 添加到结果列表
        xi_f_samples.extend(valid_xi_f.tolist())
        xi_u_samples.extend(valid_xi_u.tolist())
        
        # 显示当前进度
        if len(xi_f_samples) >= num_samples:
            break
    
    # 截取所需数量的样本
    xi_f_samples = np.array(xi_f_samples[:num_samples])
    xi_u_samples = np.array(xi_u_samples[:num_samples])
    
    # 计算有效采样率
    total_sampled = num_samples * 2  # 这只是近似值，实际会更多
    print(f"采样完成，有效采样率: {100*num_samples/(total_sampled):.1f}%")
    
    return xi_f_samples, xi_u_samples

def main():
    """主函数"""
    upper_bound = 13.5
    lower_bound = 7.5

    print("=" * 60)
    print("开始执行单链自由能优化计算")
    print(f"采样数: {num_samples}")
    print(f"参数: k={k}, Ek={Ek}, Uk={Uk}")
    print(f"网格: n_grid_points={n_grid_points}, num_r_points={r_grid_points}")
    print("=" * 60)
    
    # Step 1: 采样 ξ_f
    print(f"\nStep 1: 采样{num_samples}个ξ_f...")
    xi_f_samples, xi_u_samples = sample_xi_f_with_constraint(mu, sigma, num_samples, lower_bound, upper_bound)
    
    # 显示ξ_f和ξ_u的统计信息
    print(f"ξ_f统计: 均值={np.mean(xi_f_samples):.3f}, 标准差={np.std(xi_f_samples):.3f}")
    print(f"ξ_f范围: [{np.min(xi_f_samples):.3f}, {np.max(xi_f_samples):.3f}]")
    print(f"ξ_u统计: 均值={np.mean(xi_u_samples):.3f}, 标准差={np.std(xi_u_samples):.3f}")
    print(f"ξ_u范围: [{np.min(xi_u_samples):.3f}, {np.max(xi_u_samples):.3f}]")
    
    # 检查是否有样本不满足条件
    invalid_samples = np.sum((xi_f_samples < lower_bound) | (xi_f_samples > upper_bound))
    if invalid_samples > 0:
        print(f"警告: 有{invalid_samples}个样本不在指定范围内")
    else:
        print("所有样本均在指定范围内")
    
    # 准备n的网格
    n_grid = np.linspace(0, 1, n_grid_points)
    
    # 存储所有采样的结果
    all_r = []
    all_n = []
    all_f = []
    
    # Step 2: 对于每个采样，计算 n 和 f 随 r 的变化
    print(f"\nStep 2: 对每个采样进行优化计算...")
    
    
    for sample_idx in range(num_samples):
        xi_f = xi_f_samples[sample_idx]
        xi_u = xi_u_samples[sample_idx]
        
        # 显示进度（每10个或按比例显示）
        if sample_idx % 10 == 0:
            print(f"进度: {sample_idx+1}/{num_samples} ({100*(sample_idx+1)/num_samples:.1f}%) - ξ_f={xi_f:.3f}, ξ_u={xi_u:.3f}")
        
        # 执行优化
        r_values, n_values, f_values = optimize_for_sample(xi_f, xi_u, n_grid)
        
        # 添加到总结果中
        all_r.append(r_values)
        all_n.append(n_values)
        all_f.append(f_values)
    
    # Step 3: 保存结果到CSV文件
    print(f"\nStep 3: 保存结果...")
    save_results(all_r, all_n, all_f, xi_f_samples, xi_u_samples)
    
    # 显示统计信息
    print("\n" + "=" * 60)
    print("计算完成，统计信息:")
    print("=" * 60)
    
    # 计算并显示统计信息
    all_n_flat = np.concatenate([np.array(n) for n in all_n])
    all_f_flat = np.concatenate([np.array(f) for f in all_f])
    
    # 移除NaN值进行统计
    valid_n = all_n_flat[~np.isnan(all_n_flat)]
    valid_f = all_f_flat[~np.isnan(all_f_flat)]
    
    print(f"\nn值统计:")
    print(f"  有效点数: {len(valid_n)}/{len(all_n_flat)}")
    print(f"  范围: [{np.min(valid_n):.4f}, {np.max(valid_n):.4f}]")
    print(f"  均值: {np.mean(valid_n):.4f}")
    print(f"  标准差: {np.std(valid_n):.4f}")
    
    print(f"\nf值统计:")
    print(f"  有效点数: {len(valid_f)}/{len(all_f_flat)}")
    print(f"  范围: [{np.min(valid_f):.4f}, {np.max(valid_f):.4f}]")
    print(f"  均值: {np.mean(valid_f):.4f}")
    print(f"  标准差: {np.std(valid_f):.4f}")
    

    print("\n" + "=" * 60)
    print("所有计算完成！")
    print("=" * 60)
    print(f"结果保存在: {output_dir}")
    print("包含以下文件:")
    print("  - r_values.csv: r值数据")
    print("  - n_values.csv: n值数据")
    print("  - f_values.csv: f值数据")
    print("  - parameters.csv: 采样参数")
    print("  - config.csv: 计算配置")
    print("=" * 60)

if __name__ == "__main__":
    main()
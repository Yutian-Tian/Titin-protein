"""
系统设置：N个折叠domain串联起来的链
结构异质性：每个domain打开的能量惩罚服从高斯分布
本程序用于优化自由能函数
"""

import numpy as np
import pandas as pd
from scipy import interpolate
import os

# ==================== 参数设置 ====================
N = 5  # domain数量
Number = 100  # 采样链的数量（从300改为100）
xi_f = 10.0  # 折叠状态长度
k = 2.0  # 比例系数（根据描述 x_i u = k * x_i f）
xi_u = k * xi_f  # 展开状态长度
U0 = 20.0  # U0值（从U_0改为U0以匹配公式）
DeltaE_mean = 10.0  # DeltaE高斯分布的均值
DeltaE_std = 5.0  # DeltaE高斯分布的标准差

# 新增：采样范围限制
lower_bound = 1.0  # DeltaE的下界
upper_bound = 30.0  # DeltaE的上界

# 设置f的取值范围
f_max = 10.0  # 最大力值，可以根据需要调整
f_grid = 200  # f的网格点数
r_grid = 1000
n_grid = 200

# ==================== 指定保存路径 ====================
SAVE_DIR = "/home/tyt/project/Single-chain/opt+force/Rand_energy/results"  # 指定保存结果的目录路径

# 创建保存目录（如果不存在）
os.makedirs(SAVE_DIR, exist_ok=True)

# ==================== Step 1: 生成Energy.csv ====================
def generate_energy(save_dir=SAVE_DIR, lower_bound=lower_bound, upper_bound=upper_bound):
    print("Step 1: 生成Energy.csv...")
    print(f"采样范围: [{lower_bound}, {upper_bound}]")
    
    # 检查边界是否合理
    if lower_bound >= upper_bound:
        raise ValueError(f"下界({lower_bound})必须小于上界({upper_bound})")
    
    # 生成DeltaE数据，满足高斯分布，并在指定范围内
    # 每条链有N个domain，采样Number条链
    data = []
    
    for i in range(Number):
        # 生成N个DeltaE，满足高斯分布，确保所有值都在指定范围内
        all_in_range = False
        max_attempts = 1000  # 防止无限循环
        attempts = 0
        
        while not all_in_range and attempts < max_attempts:
            attempts += 1
            delta_es = np.random.normal(DeltaE_mean, DeltaE_std, N)
            
            # 检查所有值是否都在范围内
            if np.all((delta_es >= lower_bound) & (delta_es <= upper_bound)):
                all_in_range = True
                row = [i] + list(delta_es)
                data.append(row)
        
        # 如果超过最大尝试次数仍未生成合适的值，使用截断的方法
        if not all_in_range:
            print(f"警告: 链 {i} 在{max_attempts}次尝试后仍未生成完全在范围内的值，使用截断方法")
            # 先生成一批数据，然后将超出范围的值截断到边界
            delta_es = np.random.normal(DeltaE_mean, DeltaE_std, N)
            delta_es = np.clip(delta_es, lower_bound, upper_bound)
            row = [i] + list(delta_es)
            data.append(row)
    
    # 转换为DataFrame并保存到指定路径
    columns = ['index'] + [f'DeltaE_{j+1}' for j in range(N)]
    df = pd.DataFrame(data, columns=columns)
    save_path = os.path.join(save_dir, 'Energy.csv')
    df.to_csv(save_path, index=False)
    print(f"已生成Energy.csv，保存到: {save_path}")
    print(f"包含{Number}行数据")
    
    # 打印统计信息
    all_delta_es = np.array(data)[:, 1:].flatten().astype(float)
    print(f"DeltaE统计:")
    print(f"  最小值: {np.min(all_delta_es):.4f}")
    print(f"  最大值: {np.max(all_delta_es):.4f}")
    print(f"  均值: {np.mean(all_delta_es):.4f}")
    print(f"  标准差: {np.std(all_delta_es):.4f}")
    
    # 检查数据是否在指定范围内
    out_of_range = np.sum((all_delta_es < lower_bound) | (all_delta_es > upper_bound))
    if out_of_range > 0:
        print(f"警告: 有{out_of_range}个DeltaE值超出了指定范围[{lower_bound}, {upper_bound}]")
    else:
        print(f"所有DeltaE值都在指定范围[{lower_bound}, {upper_bound}]内")
    
    return 0

# ==================== Step 2: 读取Energy.csv并计算U_int函数 ====================
def calculate_U_int_functions(save_dir=SAVE_DIR):
    print("\nStep 2: 读取Energy.csv并计算U_int函数...")
    
    # 从指定路径读取数据
    energy_f_path = os.path.join(save_dir, 'Energy.csv')
    df = pd.read_csv(energy_f_path)
    
    # 创建插值函数列表，每个元素对应一条链的U_int插值函数
    U_int_funcs = []
    
    for idx, row in df.iterrows():
        # 获取DeltaE值（去除第一列的序号）
        delta_es = row.iloc[1:].values.astype(float)
        
        # 对DeltaE从小到大排序
        sorted_delta_es = np.sort(delta_es)
        
        # 计算整数n处的U_int值
        n_values = np.arange(N + 1)
        U_int_values = np.zeros(N + 1)
        
        # n=0时，U_int=0
        U_int_values[0] = 0
        
        # 计算前n个domain的能量之和
        for n in range(1, N + 1):
            U_int_values[n] = np.sum(sorted_delta_es[:n])
        
        # 创建插值函数，使n在[0, N]上连续
        # 使用线性插值
        U_int_func = interpolate.interp1d(n_values, U_int_values, 
                                         kind='linear', 
                                         bounds_error=False,
                                         fill_value=(U_int_values[0], U_int_values[-1]))
        U_int_funcs.append(U_int_func)
    
    print(f"已计算{len(U_int_funcs)}条链的U_int插值函数")
    return U_int_funcs

# ==================== 计算WLC自由能函数 ====================
def F_WLC(x, L_c):
    """计算WLC自由能函数，根据给定公式"""
    # 根据公式: F_WLC(x, L_c) = (π²/(2L_c)) * (1 - x²) + (2L_c)/(π(1-x²))
    # 注意：公式中的L_c可能是指持久长度，但这里按给定公式实现
    term1 = (np.pi**2 / (2 * L_c)) * (1 - x**2)
    term2 = (2 * L_c) / (np.pi * (1 - x**2))
    return term1 + term2

# ==================== 计算单链自由能 ====================
def F_c(r, n, f, U_int_func):
    """计算单链自由能 F_c(r, n; f) = F_WLC(r, n) + U(n) - f·r"""
    # 计算轮廓长度
    L_c_val = N * xi_f + n * (xi_u - xi_f)
    
    # 避免除零错误
    if L_c_val <= 0:
        return np.inf
    
    # 计算端到端因子
    x_val = r / L_c_val
    
    # 避免x>=1导致WLC公式发散
    if x_val >= 1.0 or x_val <= -1.0:
        return np.inf
    
    # 计算WLC自由能
    F_WLC_val = F_WLC(x_val, L_c_val)
    
    # 计算U(n) = U_int(n) - U0 * cos(2πn)
    U_int_val = float(U_int_func(n))
    U_val = U_int_val - U0 * np.cos(2 * np.pi * n)
    
    # 计算总自由能
    total_F = F_WLC_val + U_val - f * r
    
    return total_F

# ==================== Step 3: 计算最优(r, n)并保存结果 ====================
def calculate_optimal_r_n(U_int_funcs, save_dir=SAVE_DIR):
    print("\nStep 3: 计算最优(r, n)并保存结果...")
    
    f_values = np.linspace(0, f_max, f_grid)
    
    # 保存f_values到文件
    f_save_path = os.path.join(save_dir, 'f_values.csv')
    np.savetxt(f_save_path, f_values.reshape(-1, 1), delimiter=',', fmt='%.6f')
    print(f"已保存f_values到: {f_save_path}, 形状: {f_values.shape}")
    
    # 初始化存储数组
    # 对于每个f值，我们需要为每条链找到最优(r, n)
    # 所以最终形状是: (Number, f_grid)
    r_opt_matrix = np.zeros((Number, f_grid))
    n_opt_matrix = np.zeros((Number, f_grid))
    
    # r的取值范围
    r_max = N * xi_u
    r_values = np.linspace(0, r_max, r_grid)
    
    # n的取值范围
    n_values = np.linspace(0, N, n_grid)
    
    # 对每条链进行计算
    for chain_idx, U_int_func in enumerate(U_int_funcs):
        # 对每个f值进行计算
        for f_idx, f in enumerate(f_values):
            min_F = np.inf
            r_opt = 0
            n_opt = 0
            
            # 扫描r和n，寻找最小自由能
            for r in r_values:
                for n in n_values:
                    # 计算自由能
                    F_val = F_c(r, n, f, U_int_func)
                    
                    # 更新最小值
                    if F_val < min_F:
                        min_F = F_val
                        r_opt = r
                        n_opt = n
            
            # 存储最优值
            r_opt_matrix[chain_idx, f_idx] = r_opt
            n_opt_matrix[chain_idx, f_idx] = n_opt
        
        # 打印进度
        if (chain_idx + 1) % 10 == 0:
            print(f"  已完成{chain_idx + 1}/{len(U_int_funcs)}条链的计算")
    
    # 保存结果到CSV文件
    # 注意：根据要求，不需要任何id和表头
    r_save_path = os.path.join(save_dir, 'r_values.csv')
    n_save_path = os.path.join(save_dir, 'n_values.csv')
    
    # 保存为CSV，每列对应一个f值
    np.savetxt(r_save_path, r_opt_matrix, delimiter=',', fmt='%.6f')
    np.savetxt(n_save_path, n_opt_matrix, delimiter=',', fmt='%.6f')
    
    print(f"已保存结果到:")
    print(f"  {r_save_path}, 形状: {r_opt_matrix.shape}")
    print(f"  {n_save_path}, 形状: {n_opt_matrix.shape}")
    print(f"数据格式:")
    print(f"  f_values.csv: {f_grid}行×1列")
    print(f"  r_values.csv: {Number}行×{f_grid}列，每行对应一条链，每列对应一个f值")
    print(f"  n_values.csv: {Number}行×{f_grid}列，每行对应一条链，每列对应一个f值")
    
    return f_values, r_opt_matrix, n_opt_matrix

# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("结构异质性的影响分析程序")
    print(f"参数设置:")
    print(f"  N = {N}, Number = {Number}")
    print(f"  xi_f = {xi_f}, xi_u = {xi_u}, U0 = {U0}")
    print(f"  DeltaE_mean = {DeltaE_mean}, DeltaE_std = {DeltaE_std}")
    print(f"  采样范围: [{lower_bound}, {upper_bound}]")
    print(f"  结果保存到: {SAVE_DIR}")
    print("=" * 60)
    
    # Step 1: 生成Energy.csv
    generate_energy()
    
    # Step 2: 计算U_int函数
    U_int_funcs = calculate_U_int_functions()
    
    # Step 3: 计算最优(r, n)并保存结果
    f_values, r_opt_matrix, n_opt_matrix = calculate_optimal_r_n(U_int_funcs)
    
    print("\n" + "=" * 60)
    print("程序执行完成!")
    print("=" * 60)
    
    # 显示生成的文件
    print("\n生成的文件:")
    files_to_check = ['Energy.csv', 'f_values.csv', 'r_values.csv', 'n_values.csv']
    for file in files_to_check:
        file_path = os.path.join(SAVE_DIR, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"  {file}: {size:.2f} KB")
    
    # 显示部分结果
    print(f"\n结果预览:")
    print(f"  f_values范围: {f_values[0]:.2f} 到 {f_values[-1]:.2f}")
    print(f"  r_opt范围: {np.min(r_opt_matrix):.2f} 到 {np.max(r_opt_matrix):.2f}")
    print(f"  n_opt范围: {np.min(n_opt_matrix):.2f} 到 {np.max(n_opt_matrix):.2f}")

# ==================== 执行程序 ====================
if __name__ == "__main__":
    main()
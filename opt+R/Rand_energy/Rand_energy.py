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
Number = 300  # 采样链的数量
xi_f = 10.0  # 折叠状态长度
k = 2.0  # 比例系数（根据描述 x_i u = k * x_i f）
xi_u = k*xi_f  # 展开状态长度
U0 = 20.0  # U0值
DeltaE_mean = 10.0  # DeltaE高斯分布的均值
DeltaE_std = 5.0  # DeltaE高斯分布的标准差

# 新增：采样范围限制
lower_bound = 1.0  # DeltaE的下界
upper_bound = 30.0  # DeltaE的上界

r_grid = 1000
n_grid = 200

# ==================== 指定保存路径 ====================
SAVE_DIR = "/home/tyt/project/Single-chain/opt+R/Rand_energy/results"  # 指定保存结果的目录路径

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

# ==================== Step 3: 计算n_opt和f值 ====================
def calculate_n_opt_and_f(U_int_funcs, r_grid, n_grid, save_dir=SAVE_DIR):
    print("\nStep 3: 计算n_opt和f值...")
    
    # 计算r的取值范围
    r_max = 0.95 * N * xi_u
    r_values = np.linspace(0, r_max, r_grid)  # 划分200个网格点
    
    # 初始化存储数组
    all_n_opts = []
    all_f_values = []
    
    # 对每条链进行计算
    for chain_idx, U_int_func in enumerate(U_int_funcs):
        n_opts = []
        f_vals = []
        
        # 对每个r值计算最优n和对应的力f
        for r in r_values:
            # 在n属于[0, N]范围内寻找使自由能最小的n
            # 使用离散化搜索
            n_grid_vals = np.linspace(0, N, n_grid)  # 离散化n
            
            # 计算自由能F_c(r, n)
            F_c_values = []
            
            for n in n_grid_vals:
                # 计算L_c(n)
                L_c = N * xi_f + n * (xi_u - xi_f)
                
                # 避免除零错误
                if L_c <= 0:
                    F_c_values.append(np.inf)
                    continue
                
                # 计算x = r / L_c(n)
                x = r / L_c
                
                # 避免x>=1导致WLC公式发散
                if x >= 0.99:
                    F_c_values.append(np.inf)
                    continue
                
                # 计算F_WLC(x, L_c)
                term1 = (np.pi**2) / (2 * L_c**2) * (1 - x**2)
                term2 = (2 * L_c) / (np.pi * (1 - x**2))
                F_WLC = term1 + term2
                
                # 计算U(n)
                U_int = float(U_int_func(n))  # 调用插值函数
                U_n = U_int - U0 * np.cos(2 * np.pi * n)
                
                # 计算总自由能
                F_c = F_WLC + U_n
                F_c_values.append(F_c)
            
            # 找到最小自由能对应的n
            if len(F_c_values) > 0 and np.min(F_c_values) < np.inf:
                min_idx = np.argmin(F_c_values)
                n_opt = n_grid_vals[min_idx]
                
                # 计算对应的力f
                L_c_opt = N * xi_f + n_opt * (xi_u - xi_f)
                x_opt = r / L_c_opt if L_c_opt > 0 else 0
                
                # 使用给定的力公式
                if x_opt < 0.99:  # 避免发散
                    term1 = - (np.pi**2 * x_opt) / (L_c_opt**2)
                    term2 = (4 * x_opt) / (np.pi * (1 - x_opt**2)**2)
                    f = term1 + term2
                else:
                    f = 0
            else:
                n_opt = 0
                f = 0
            
            n_opts.append(n_opt)
            f_vals.append(f)
        
        all_n_opts.append(n_opts)
        all_f_values.append(f_vals)
        
        if (chain_idx + 1) % 20 == 0:
            print(f"  已完成{chain_idx + 1}/{len(U_int_funcs)}条链的计算")
    
    # 保存结果到指定路径的CSV文件
    # 保存r值
    r_save_path = os.path.join(save_dir, 'r_values.csv')
    r_df = pd.DataFrame(r_values)
    r_df.to_csv(r_save_path, index=False, header=False)
    
    # 保存n_opt值 - 第一行不要标签，第一列不要存储r_values的数据
    n_save_path = os.path.join(save_dir, 'n_values.csv')
    n_data = np.array(all_n_opts).T  # 转置，使每行对应一个r值，每列对应一条链
    np.savetxt(n_save_path, n_data, delimiter=',')
    
    # 保存f值 - 第一行不要标签，第一列不要存储r_values的数据
    f_save_path = os.path.join(save_dir, 'f_values.csv')
    f_data = np.array(all_f_values).T  # 转置，使每行对应一个r值，每列对应一条链
    np.savetxt(f_save_path, f_data, delimiter=',')
    
    print(f"已保存结果到:")
    print(f"  {r_save_path}")
    print(f"  {n_save_path}")
    print(f"  {f_save_path}")
    print(f"数据格式:")
    print(f"  r_values.csv: {r_grid}行×1列，无表头")
    print(f"  n_values.csv: {r_grid}行×{Number}列，无表头，每行对应一个r值，每列对应一条链")
    print(f"  f_values.csv: {r_grid}行×{Number}列，无表头，每行对应一个r值，每列对应一条链")
    
    return r_values, np.array(all_n_opts), np.array(all_f_values)

# ==================== 主程序 ====================
def main():
    print("=" * 50)
    print("结构异质性的影响分析程序")
    print(f"结果将保存到: {SAVE_DIR}")
    print(f"采样范围: [{lower_bound}, {upper_bound}]")
    print("=" * 50)
    
    # Step 1: 生成Energy.csv
    cha = generate_energy()
    
    # Step 2: 计算U_int函数
    U_int_funcs = calculate_U_int_functions()
    
    # Step 3: 计算n_opt和f值
    r_values, all_n_opts, all_f_values = calculate_n_opt_and_f(U_int_funcs, r_grid, n_grid)
    
    print("\n" + "=" * 50)
    print("程序执行完成!")
    print("=" * 50)
    
    # 显示生成的文件
    print("\n生成的文件:")
    files_to_check = ['Energy.csv', 'r_values.csv', 'n_values.csv', 'f_values.csv']
    for file in files_to_check:
        file_path = os.path.join(SAVE_DIR, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"  {file_path}: {size:.2f} KB")

# ==================== 执行程序 ====================
if __name__ == "__main__":
    main()
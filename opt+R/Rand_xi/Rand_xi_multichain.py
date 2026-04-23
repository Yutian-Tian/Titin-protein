"""
系统设置：N个折叠domain串联起来的链
结构异质性：折叠长度xi_f服从高斯分布
修改：对每个链的xi_f从小到大排序，xi_f小的domain先打开
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import warnings
import os

warnings.filterwarnings('ignore')

# ==================== 参数设置 ====================
# 问题中给出的参数
U0 = 2.1
Number = 100         # 链的数目
N = 10                # 每条链的domain数目

# 需要补充的参数（问题未给出，设为合理默认值）
xi_f_mean = 9.0     # xi_f高斯分布的均值
xi_f_std = 3.0       # xi_f高斯分布的标准差
k = 2.0              # 展开长度与折叠长度的比例：xi_u = k * xi_f

# 采样范围约束
lower_bound = 5.0    # xi_f采样的下界
upper_bound = 13.0   # xi_f采样的上界

# 数值计算参数
r_grid = 500        # r网格点数
n_grid = 500         # n离散点数，用于搜索最小值

# 指定保存路径
SAVE_PATH = "/home/tyt/project/Single-chain/opt+R/Rand_xi/simulation_results"  # 可以修改为任意指定路径

# 确保保存路径存在
os.makedirs(SAVE_PATH, exist_ok=True)

# ==================== Step 1: 生成xi_f样本并保存 ====================
def generate_xi_f():
    """生成Number组xi_f样本，每组N个，服从高斯分布，保存为xi_f.csv"""
    
    print("正在生成xi_f样本数据...")
    
    # 生成所有样本数据，确保每个xi_f在[lower_bound, upper_bound]范围内
    xi_f_samples = np.zeros((Number, N))
    
    for chain_idx in range(Number):
        # 为每条链生成N个xi_f值
        chain_values = np.zeros(N)
        for i in range(N):
            # 循环生成直到获得指定范围内的值
            value = 0
            while value <= 0 or value < lower_bound or value > upper_bound:
                value = np.random.normal(xi_f_mean, xi_f_std)
            chain_values[i] = value
        
        # 存储这条链的数据
        xi_f_samples[chain_idx] = chain_values
    
    # 添加序号列（第一列）
    data_with_index = np.hstack((np.arange(Number).reshape(-1, 1), xi_f_samples))
    
    # 创建DataFrame并保存到指定路径
    columns = ['index'] + [f'xi_f_{i}' for i in range(N)]
    df = pd.DataFrame(data_with_index, columns=columns)
    
    # 保存到指定路径
    xi_f_path = os.path.join(SAVE_PATH, 'xi_f.csv')
    df.to_csv(xi_f_path, index=False)
    
    # 验证所有数据都在指定范围内
    if np.any(df.iloc[:, 1:] <= 0):
        print("警告：存在非正值数据！")
    elif np.any(df.iloc[:, 1:] < lower_bound) or np.any(df.iloc[:, 1:] > upper_bound):
        print("警告：存在超出范围的数据！")
    else:
        print(f"Step 1完成：已生成{Number}条链的xi_f数据，所有值均在[{lower_bound}, {upper_bound}]范围内")
        print(f"  保存至: {xi_f_path}")
    
    return df, xi_f_path

# ==================== Step 2: 读取xi_f，排序后计算L_c(n)和U_int(n) ====================
def compute_interpolations(xi_f):
    """对单条链计算L_c(n)和U_int(n)的插值函数"""
    # 计算每个domain的展开长度
    xi_u = k * xi_f
    
    # 计算每个domain的能量DeltaEi
    DeltaEi = 0.1 + 0.5 * (xi_f - 5.0)
    
    # 整数点n=0,...,N
    n_int = np.arange(N + 1)  # [0,1,...,N]
    
    # 计算整数点的L_c(n)
    L_c_int = np.zeros(N + 1)
    for i, n in enumerate(n_int):
        # 前n个domain展开（xi_f小的先打开），后(N-n)个折叠
        L_c_int[i] = np.sum(xi_u[:n]) + np.sum(xi_f[n:])
    
    # 计算整数点的U_int(n)
    U_int_n = np.zeros(N + 1)
    for i, n in enumerate(n_int):
        if n == 0:
            U_int_n[i] = 0.0
        else:
            U_int_n[i] = np.sum(DeltaEi[:n])
    
    # 创建线性插值函数
    L_c_func = interp1d(n_int, L_c_int, kind='linear',
                        bounds_error=False, fill_value=(L_c_int[0], L_c_int[-1]))
    U_int_func = interp1d(n_int, U_int_n, kind='linear',
                          bounds_error=False, fill_value=(U_int_n[0], U_int_n[-1]))
    
    return L_c_func, U_int_func, L_c_int, DeltaEi

# ==================== WLC自由能计算函数 ====================
def F_WLC(x, L_c):
    """计算WLC自由能，根据图中公式"""
    # 将输入转换为数组以确保一致性
    x = np.asarray(x)
    L_c = np.asarray(L_c)
    
    # 初始化结果数组
    result = np.zeros_like(x, dtype=float)
    
    # 处理x < 1的情况
    mask = x < 1.0
    if np.any(mask):
        x_masked = x[mask]
        L_c_masked = L_c[mask]
        
        term1 = (np.pi**2 / (2.0 * L_c_masked)) * (1.0 - x_masked**2)
        term2 = (2.0 * L_c_masked) / (np.pi * (1.0 - x_masked**2))
        
        result[mask] = term1 + term2
    
    # 处理x >= 1的情况
    result[~mask] = np.inf
    
    # 如果输入是标量，则返回标量
    if result.size == 1:
        return float(result)
    
    return result

def F_MS(x, L_c):
    """计算WLC自由能，根据图中公式"""
    # 将输入转换为数组以确保一致性
    x = np.asarray(x)
    L_c = np.asarray(L_c)
    
    # 初始化结果数组
    result = np.zeros_like(x, dtype=float)
    
    # 处理x < 1的情况
    mask = x < 1.0
    if np.any(mask):
        x_masked = x[mask]
        L_c_masked = L_c[mask]
        result[mask] = 0.25 * L_c_masked * x_masked**2 * (3.0 - 2.0 * x_masked)/(1.0 - x_masked)
    
    # 处理x >= 1的情况
    result[~mask] = np.inf
    
    # 如果输入是标量，则返回标量
    if result.size == 1:
        return float(result)
    
    return result

def force_WLC(x, L_c):
    """计算WLC力，根据图中公式"""
    # 将输入转换为数组以确保一致性
    x = np.asarray(x)
    L_c = np.asarray(L_c)
    
    # 初始化结果数组
    result = np.zeros_like(x, dtype=float)
    
    # 处理x < 1的情况
    mask = x < 1.0
    if np.any(mask):
        x_masked = x[mask]
        L_c_masked = L_c[mask]
        
        term1 = - (np.pi**2 * x_masked) / (L_c_masked**2)
        term2 = (4 * x_masked) / (np.pi * (1 - x_masked**2)**2)

        result[mask] = term1 + term2
    
    # 处理x >= 1的情况
    result[~mask] = np.inf
    
    # 如果输入是标量，则返回标量
    if result.size == 1:
        return float(result)
    
    return result

def force_MS(x, L_c):
    """计算WLC力，根据图中公式"""
    # 将输入转换为数组以确保一致性
    x = np.asarray(x)
    L_c = np.asarray(L_c)
    
    # 初始化结果数组
    result = np.zeros_like(x, dtype=float)
    
    # 处理x < 1的情况
    mask = x < 1.0
    if np.any(mask):
        x_masked = x[mask]
        result[mask] = 0.25 * ((1 - x_masked)**(-2) -1 + 4*x_masked)
    
    # 处理x >= 1的情况
    result[~mask] = np.inf
    
    # 如果输入是标量，则返回标量
    if result.size == 1:
        return float(result)
    
    return result

# ==================== Step 3: 对每个r扫描自由能，找到最优n ====================
def compute_optimal_n_for_chain(xi_f):
    """对单条链计算每个r对应的最优n和力"""
    # 获取插值函数
    L_c_func, U_int_func, L_c_int, DeltaE= compute_interpolations(xi_f)
    
    # 生成r网格：从0到最大轮廓长度L_c(N)
    r_max = float(L_c_func(N))
    r_values = np.linspace(0, 0.95 * r_max, r_grid)  # 避免x太接近1
    
    # 离散化n用于搜索最小值
    n_values = np.linspace(0, N, n_grid)
    
    # 存储结果
    n_opt_values = np.zeros_like(r_values)
    f_values = np.zeros_like(r_values)
    
    # 对每个r值扫描自由能
    for i, r in enumerate(r_values):
        # 计算每个n对应的自由能
        F_values = np.zeros_like(n_values)
        
        for j, n in enumerate(n_values):
            # 计算L_c(n)和x
            L_c_n = float(L_c_func(n))
            
            if r > L_c_n:  # r不能超过轮廓长度
                F_values[j] = np.inf
                continue
                
            x = r / L_c_n
            
            if x >= 1:  # 物理上不允许
                F_values[j] = np.inf
                continue
            
            # 计算WLC自由能
            #F_wlc = F_WLC(x, L_c_n)
            F_wlc = F_MS(x, L_c_n)
            
            # 计算U(n)项
            # U_n = float(U_int_func(n)) - DeltaE[int(n - 0.1)] * np.cos(2.0 * np.pi * n)
            U_n = float(U_int_func(n)) - U0 * np.cos(2.0 * np.pi * n)
            
            # 总自由能
            F_values[j] = F_wlc + U_n
        
        # 找到最小自由能对应的n
        if np.all(np.isinf(F_values)):
            n_opt = 0
        else:
            min_idx = np.nanargmin(F_values)
            n_opt = n_values[min_idx]
        
        # 计算对应的力
        L_c_opt = float(L_c_func(n_opt))
        x_opt = r / L_c_opt
        
        if x_opt >= 1:
            f_val = np.inf
        else:
            #f_val = force_WLC(x_opt, L_c_opt)
            f_val = force_MS(x_opt, L_c_opt)
        
        n_opt_values[i] = n_opt
        f_values[i] = f_val
    
    return r_values, n_opt_values, f_values

# ==================== 主计算函数 ====================
def compute_all_chains(xi_f_path):
    """读取xi_f.csv，对所有链进行计算，保存结果"""
    print("\n开始计算每条链的r、n和f值...")
    
    # 从指定路径读取数据
    df_xi_f = pd.read_csv(xi_f_path)
    xi_f_all = df_xi_f.iloc[:, 1:].values.astype(float)  # 忽略第一列序号
    
    # 对每条链的xi_f进行从小到大排序（修改部分）
    print("对每条链的xi_f进行从小到大排序...")
    xi_f_sorted = np.sort(xi_f_all, axis=1)  # 默认就是从小到大排序
    
    # 验证数据
    if np.any(xi_f_sorted <= 0):
        print("警告：排序后的xi_f中存在非正值数据！")
        # 将非正值替换为很小的正数以避免计算错误
        xi_f_sorted[xi_f_sorted <= 0] = 1e-6
    
    # 存储所有链的结果
    all_r = []
    all_n = []
    all_f = []
    
    print(f"  预计计算 {Number} 条链...")
    
    for idx in range(Number):
        xi_f = xi_f_sorted[idx]  # 使用排序后的xi_f
        r, n_opt, f = compute_optimal_n_for_chain(xi_f)
        
        all_r.append(r)
        all_n.append(n_opt)
        all_f.append(f)
        
        # 每10条链输出一次进度
        if (idx + 1) % 10 == 0:
            print(f"  已完成 {idx + 1}/{Number} 条链的计算")
    
    # 转换为DataFrame（每列一条链，每行一个r网格点）
    df_r = pd.DataFrame(np.array(all_r).T)
    df_n = pd.DataFrame(np.array(all_n).T)
    df_f = pd.DataFrame(np.array(all_f).T)
    
    # 保存为CSV文件到指定路径
    r_value_path = os.path.join(SAVE_PATH, 'r_values.csv')
    n_value_path = os.path.join(SAVE_PATH, 'n_values.csv')
    f_value_path = os.path.join(SAVE_PATH, 'f_values.csv')
    
    df_r.to_csv(r_value_path, index=False, header=False)
    df_n.to_csv(n_value_path, index=False, header=False)
    df_f.to_csv(f_value_path, index=False, header=False)
    
    print(f"\nStep 2&3完成：结果已保存至以下路径：")
    print(f"  r值数据: {r_value_path}")
    print(f"  n值数据: {n_value_path}")
    print(f"  力值数据: {f_value_path}")
    
    return df_r, df_n, df_f

# ==================== 可选：添加统计信息输出 ====================
def print_statistics(xi_f_path):
    """打印采样数据的统计信息"""
    df = pd.read_csv(xi_f_path)
    xi_f_data = df.iloc[:, 1:].values
    
    print("\n" + "="*40)
    print("原始采样数据统计信息")
    print("="*40)
    print(f"数据文件: {xi_f_path}")
    print(f"数据形状: {xi_f_data.shape}")
    print(f"所有数据是否大于0: {np.all(xi_f_data > 0)}")
    print(f"所有数据是否在[{lower_bound}, {upper_bound}]范围内: {np.all((xi_f_data >= lower_bound) & (xi_f_data <= upper_bound))}")
    print(f"最小值: {np.min(xi_f_data):.6f}")
    print(f"最大值: {np.max(xi_f_data):.6f}")
    print(f"均值: {np.mean(xi_f_data):.6f}")
    print(f"标准差: {np.std(xi_f_data):.6f}")
    print("="*40)

# ==================== 主程序 ====================
if __name__ == '__main__':
    print("="*60)
    print("开始计算结构异质性的影响")
    print("注意：对每个链的xi_f从小到大排序，xi_f小的domain先打开")
    print("="*60)
    print(f"结果将保存到指定路径: {SAVE_PATH}")
    print(f"参数设置:")
    print(f"  链的数目 (Number): {Number}")
    print(f"  每条链的domain数目 (N): {N}")
    print(f"  展开长度比例 (k): {k}")
    print(f"  采样范围: [{lower_bound}, {upper_bound}]")
    print(f"  U0: {U0}")
    print("="*60)
    
    # Step 1: 生成xi_f样本（确保所有值在指定范围内）
    df_xi_f, xi_f_path = generate_xi_f()
    
    # Step 2 & 3: 计算所有链（包含排序步骤）
    compute_all_chains(xi_f_path)
    
    # 打印统计信息
    print_statistics(xi_f_path)
    
    print("\n" + "="*60)
    print("所有计算完成！")
    print(f"所有文件已保存到: {SAVE_PATH}")
    
    # 显示生成的文件列表
    files = os.listdir(SAVE_PATH)
    print("\n生成的文件列表:")
    for file in sorted(files):
        if file.endswith('.csv'):
            file_path = os.path.join(SAVE_PATH, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"  - {file} ({file_size:.2f} KB)")
    
    print("="*60)
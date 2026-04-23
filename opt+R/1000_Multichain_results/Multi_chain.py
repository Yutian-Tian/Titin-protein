import numpy as np
import pandas as pd


def sample_from_gaussian(mu, sigma, N, filepath):
    # 使用numpy生成正态分布随机数
    samples1 = np.random.normal(mu, sigma, N)
    samples2 = np.array(samples1)
    samples3 = np.sort(samples2)

    
    # 将结果追加到文件（每次取样作为一行）
    with open(filepath, 'a') as f:
        # 将数组转换为字符串，并用空格分隔
        samples_str = ' '.join([f'{x:.6f}' for x in samples3])
        f.write(samples_str + '\n')  # 添加换行符，确保每次取样在新的一行
    return samples3


def optimization_function(a, U0, L, xi, txt_filename, line_number):
    
    r_points = 1000
    scan_points = 1000
    
    # 读取指定行数据
    with open(txt_filename, 'r') as f:
        lines = f.readlines()
        line = lines[line_number].strip()  # 读取第i行
    
    # 将字符串转换为numpy数组
    data1 = np.array([float(x) for x in line.split()])
    data2 = np.linspace(0, 9, 10)
    data3 = np.cumsum(data1)
    
    # n_u的取值范围
    n_min = 0.0
    n_max = float(len(data2))  # 根据数据长度调整
    

    def F(n, r):
        c_val = 1 / (2 * (L - (n_max - n) * xi))
        x_val = (r - (n_max - n) * a) / (L - (n_max - n) * xi)
        denominator = 1 - x_val**2
        return np.pi**2 * c_val * denominator + 1 / (1e-16 + np.pi * c_val * denominator)\
        + np.interp(n, data2, data3) - U0 * np.cos(2 * np.pi * n)
    
    # 计算F对r的精确偏导数
    def exact_partial_F_r(n, r):
        # 计算中间变量
        Lc = L - (n_max - n) * xi
        x_val = (r - (n_max - n) * a) / Lc
        
        # 计算导数
        d_term1_dr = - np.pi**2 * x_val / Lc**2
        d_term2_dr = 4 * x_val / (np.pi * (1 - x_val**2)**2)
        
        # U(n)不依赖于r，其导数为0
        return d_term1_dr + d_term2_dr
    
    # 生成r的取值范围
    r_max = L - n_min * xi - 50.0
    r_values = np.linspace(n_max * a, r_max, r_points)
    
    # 存储结果
    optimal_ns = []
    min_F_values = []
    partial_F_r_values = []  # 存储偏导数值
    
#    print("开始均匀扫描优化并计算偏导数...")
#    print(f"使用数据点: {data1}")
#    print(f"n_max: {n_max}, r_points: {r_points}")
    
    # 对每个r值进行均匀扫描优化并计算偏导数
    for i, r in enumerate(r_values):
        # 对于给定的r，计算n的实际取值范围
        n_lower_bound = max(n_min, n_max - (L - r) / xi)
        
        # 如果下界大于上界，跳过
        if n_lower_bound >= n_max:
            optimal_ns.append(np.nan)
            min_F_values.append(np.nan)
            partial_F_r_values.append(np.nan)
            continue
        
        # 在n的取值范围内均匀采样
        n_samples = np.linspace(n_lower_bound, n_max, scan_points)
        F_samples = []
        
        # 计算每个采样点的F值
        for n in n_samples:
            try:
                F_val = F(n, r)
                F_samples.append(F_val)
            except:
                F_samples.append(np.inf)
        
        F_samples = np.array(F_samples)
        
        # 找到最小F值对应的n
        min_index = np.nanargmin(F_samples)
        min_F = F_samples[min_index]
        optimal_n = n_samples[min_index]
        
        optimal_ns.append(optimal_n)
        min_F_values.append(min_F)
        
        # 使用精确导数计算偏导数
        try:
            dF_dr = exact_partial_F_r(optimal_n, r)
            partial_F_r_values.append(dF_dr)
        except:
            partial_F_r_values.append(np.nan)
            if i % 1000 == 0:  # 减少输出频率
                print(f"警告: 在r={r:.3f}, nf={optimal_n:.6f}处计算偏导数失败")
        
        # 显示进度
        # if (i + 1) % 2000 == 0:
        #     print(f"处理进度: {i + 1}/{r_points}, r={r:.3f}, 最优nf={optimal_n:.6f}, 最小F={min_F:.6f}, ∂F/∂r={partial_F_r_values[-1]:.6f}")
    
    # 转换为numpy数组
    optimal_ns = np.array(optimal_ns)
    min_F_values = np.array(min_F_values)
    partial_F_r_values = np.array(partial_F_r_values)
    eefactor = r_values/(L-(n_max-optimal_ns)*xi)
    
    # 创建一个结果字典，包含所有重要结果
    result = {
        'r_values': r_values,
        'optimal_ns': optimal_ns,
        'min_F_values': min_F_values,
        'force': partial_F_r_values,
        'eefactor': eefactor,
        'input_data': data1,
        'n_max': n_max
    }
    
#    print("优化完成!")
#    print(f"r_values范围: {r_values[0]:.3f} 到 {r_values[-1]:.3f}")
#    print(f"optimal_ns范围: {np.nanmin(optimal_ns):.3f} 到 {np.nanmax(optimal_ns):.3f}")
#    print(f"min_F_values范围: {np.nanmin(min_F_values):.3f} 到 {np.nanmax(min_F_values):.3f}")
    
    return result

def save_result_to_csv(result, filepath, name):
    # 创建新的force数据列
    new_force_data = result[name]
    
    try:
        # 尝试读取已存在的文件
        df = pd.read_csv(filepath)
        
        # 为新列生成唯一的列名（force_1, force_2, ...）
        col_number = 1
        new_col_name = 'force'
        
        while new_col_name in df.columns:
            col_number += 1
            new_col_name = f'force_{col_number}'
        
        # 添加新列
        df[new_col_name] = new_force_data
        
        # 保存回文件
        df.to_csv(filepath, index=False)
        
    except FileNotFoundError:
        # 如果文件不存在，创建新文件
        df_data = {
            'force': new_force_data  # 第一列名为 'force'
        }
        df = pd.DataFrame(df_data)
        df.to_csv(filepath, index=False)
        
        print(f"✅ 创建新文件 {filepath}，并添加 'force' 列")


def main():

    # 参数设置
    a = 0.0
    U0 = 10.0
    L = 400.0
    xi = 30.0
    Num = 1000 # 运行100次

    # 取样参数
    mu = 10.0
    sigma = 3.0
    # 存储路径
    Etxt_filepath = "/home/tyt/project/Single-chain/opt+R/1000_Multichain_results/energy.txt"
    fcsv_filepath = "/home/tyt/project/Single-chain/opt+R/1000_Multichain_results/f_multichain.csv"
    ncsv_filepath = "/home/tyt/project/Single-chain/opt+R/1000_Multichain_results/n_multichain.csv"

    for i in range(Num):
        if (i+1) % 10 == 0:
            print(f'处理进度: {i + 1}/{Num}')
        sample_from_gaussian(mu, sigma, 10, Etxt_filepath)
        result = optimization_function(a, U0, L, xi, Etxt_filepath, i)

        save_result_to_csv(result, fcsv_filepath, name='force')
        save_result_to_csv(result, ncsv_filepath, name='optimal_ns')
    
    print('Process Completed.')
    return 0

if __name__ == "__main__":
    main()
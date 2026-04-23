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


def filter_and_save_positive_samples(input_filepath, output_filepath, start_line=0, end_line=None):

    valid_indices = []  # 有效行在原始文件中的行号
    valid_count = 0
    invalid_count = 0
    
    # 读取输入文件
    with open(input_filepath, 'r') as f:
        lines = f.readlines()
    
    # 设置结束行号
    if end_line is None:
        end_line = len(lines)
    
    # 确保行号在有效范围内
    start_line = max(0, start_line)
    end_line = min(len(lines), end_line)
    
    # 清空或创建输出文件
    with open(output_filepath, 'w') as f:
        f.write("")  # 创建空文件或清空已有内容
    
    print(f"处理行范围: {start_line} 到 {end_line-1}")
    
    # 遍历指定行范围
    for i in range(start_line, end_line):
        line = lines[i].strip()
        if not line:  # 跳过空行
            continue
            
        # 将字符串转换为numpy数组
        try:
            data = np.array([float(x) for x in line.split()])
        except ValueError:
            print(f"警告: 第{i}行数据格式错误，跳过")
            invalid_count += 1
            continue
        
        # 检查是否所有值均为正数
        if np.all(data > 0):
            valid_count += 1
            valid_indices.append(i)  # 记录原始文件中的行号
            
            # 将有效数据写入输出文件
            with open(output_filepath, 'a') as f:
                samples_str = ' '.join([f'{x:.6f}' for x in data])
                f.write(samples_str + '\n')
        else:
            invalid_count += 1
    
    # 输出统计信息
    total_processed = end_line - start_line
    print(f"\n筛选完成！")
    print(f"处理总行数: {total_processed}")
    print(f"有效数据行数: {valid_count}")
    print(f"有效数据已保存到: {output_filepath}")

    
    return valid_indices, valid_count, invalid_count


def optimization_function(a, U0, L, xi, txt_filename, line_number):
    
    r_points = 1000
    scan_points = 1000
    
    # 读取指定行数据
    with open(txt_filename, 'r') as f:
        lines = f.readlines()
        line = lines[line_number].strip()  # 读取第i行
    
    # 将字符串转换为numpy数组
    data1 = np.array([float(x) for x in line.split()])
    data2 = np.linspace(0, 10, 11)
    data = np.insert(data1, 0, 0.0)
    data3 = np.cumsum(data)
    
    # n_u的取值范围
    n_min = 0.0
    n_max = float(len(data2)-1)  # 根据数据长度调整
    

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
    L = 600.0
    xi = 50.0
    Num = 100000 # 运行100次

    # 取样参数
    mu = 10.0
    sigma = 100.0
    # 存储路径
    Etxt_filepath = "/home/tyt/project/Single-chain/opt+R/Multichain_results/energy.txt"
    energy_eff_filepath = "/home/tyt/project/Single-chain/opt+R/Multichain_results/energy_eff.txt"
    fcsv_filepath = "/home/tyt/project/Single-chain/opt+R/Multichain_results/f_values.csv"
    ncsv_filepath = "/home/tyt/project/Single-chain/opt+R/Multichain_results/n_values.csv"
    rcsv_filepath = "/home/tyt/project/Single-chain/opt+R/Multichain_results/r_values.csv"

    # sample
    for i in range(Num):
        sample_from_gaussian(mu, sigma, 10, Etxt_filepath)

    valid_indices, valid_count, invalid_count = filter_and_save_positive_samples(
        input_filepath=Etxt_filepath,
        output_filepath=energy_eff_filepath
    )

    for i, original_line_num in enumerate(valid_indices):
        if (i+1) % 10 == 0:
            print(f'  优化计算进度: {i + 1}/{valid_count}')
        
        # 使用筛选后的文件进行优化计算
        # 注意：exergy_eff.txt中的第i行对应原始文件的第original_line_num行
        result = optimization_function(a, U0, L, xi, energy_eff_filepath, i)
        
        # 保存结果
        save_result_to_csv(result, fcsv_filepath, name='force')
        save_result_to_csv(result, ncsv_filepath, name='optimal_ns')
    
    save_result_to_csv(result, rcsv_filepath, name='r_values')
    print('Process Completed.')
    return 0

if __name__ == "__main__":
    main()
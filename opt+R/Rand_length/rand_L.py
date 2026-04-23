import numpy as np
import os


def optimization_function(U0, L, xi_u, xi_f, data):
    
    r_points = 2000
    scan_points = 1000   
    
    # 数值精度控制参数
    EPSILON = 1e-14  # 更小的epsilon用于避免除零
    X_THRESHOLD = 0.999  # x_val的阈值，避免分母过小
    
    # 将字符串转换为numpy数组
    data1 = np.array(data)
    data1 = np.insert(data1, 0, 0.0)
    data2 = np.linspace(0, 10, 11)
    data3 = np.cumsum(data1)
    
    # n_u的取值范围
    n_min = 0.0
    n_max = float(len(data2) - 1)  # 根据数据长度调整
    
    # 预计算插值函数，提高效率
    # 创建线性插值函数
    from scipy.interpolate import interp1d
    interp_func = interp1d(data2, data3, kind='linear', fill_value='extrapolate')
    
    # 优化的U函数
    def U(n):
        # 使用预计算的插值函数
        return interp_func(n) - U0 * np.cos(2 * np.pi * n)
    
    # 数值稳定的F函数
    def F(n, r):
        # 计算中间变量
        Lc = L - (n_max - n) * (xi_u - xi_f)
        
        # 检查Lc是否为正，避免负数或零
        if Lc <= EPSILON:
            return np.inf
        
        # 计算c值和x值
        c_val = 1.0 / (2.0 * Lc)
        x_val = r / Lc
        
        # 确保x_val在合理范围内，避免分母为零
        # 使用更严格的条件限制x_val
        if np.abs(x_val) >= X_THRESHOLD:
            # 当x_val接近1时，使用近似计算避免数值问题
            x_val = np.sign(x_val) * X_THRESHOLD
        
        denominator = 1.0 - x_val**2
        
        # 保护分母，避免为零
        if denominator <= EPSILON:
            denominator = EPSILON
        
        # 计算两项，注意数值稳定性
        term1 = np.pi**2 * c_val * denominator
        
        # 第二项的分母也需要保护
        term2_denominator = np.pi * c_val * denominator
        if term2_denominator <= EPSILON:
            term2_denominator = EPSILON
        
        term2 = 1.0 / term2_denominator
        
        return term1 + term2 + U(n)
    
    # 计算F对r的精确偏导数（优化版本）
    def exact_partial_F_r(n, r):
        # 计算中间变量
        Lc = L - (n_max - n) * (xi_u - xi_f)
        
        # 检查Lc是否为正
        if Lc <= EPSILON:
            return 0.0
        
        x_val = r / Lc
        
        # 确保x_val在合理范围内
        if np.abs(x_val) >= X_THRESHOLD:
            x_val = np.sign(x_val) * X_THRESHOLD
        
        denominator = 1.0 - x_val**2
        
        # 保护分母
        if denominator <= EPSILON:
            denominator = EPSILON
        
        # 计算c值
        c_val = 1.0 / (2.0 * Lc)
        
        # 计算导数项，使用数值更稳定的公式
        # 第一项导数: d(π²c(1-x²))/dr = -2π²c x / Lc
        d_term1_dr = -2.0 * np.pi**2 * c_val * x_val / Lc
        
        # 第二项导数: d(1/(πc(1-x²)))/dr = 2x / (πc Lc (1-x²)²)
        # 使用更稳定的计算方式
        term2_factor = 2.0 * x_val / (np.pi * c_val * Lc)
        d_term2_dr = term2_factor / (denominator**2)
        
        return d_term1_dr + d_term2_dr
    
    # 生成r的取值范围
    r_max = L
    r_min = 0.0
    
    # 确保r_max > r_min
    if r_max <= r_min:
        r_max = r_min + 1.0
    
    r_values = np.linspace(r_min, 0.95*r_max, r_points)
    
    # 预分配数组
    optimal_ns = np.full(r_points, np.nan)
    min_F_values = np.full(r_points, np.nan)
    partial_F_r_values = np.full(r_points, np.nan)
    
    # 对每个r值进行均匀扫描优化并计算偏导数
    for i, r in enumerate(r_values):
        # 对于给定的r，计算n的实际取值范围
        n_lower_bound = max(n_min, n_max - (L - r) / (xi_u - xi_f))
        
        # 如果下界大于上界，跳过
        if n_lower_bound >= n_max:
            continue
        
        # 在n的取值范围内均匀采样
        n_samples = np.linspace(n_lower_bound, n_max, scan_points)
        
        # 使用向量化计算提高效率
        F_samples = np.array([F(n, r) for n in n_samples])
        
        # 找到最小F值对应的n
        valid_mask = np.isfinite(F_samples)
        if not np.any(valid_mask):
            continue
        
        min_index = np.argmin(F_samples[valid_mask])
        # 注意：min_index是有效值中的索引，需要映射回原始索引
        original_indices = np.where(valid_mask)[0]
        min_index_original = original_indices[min_index]
        
        min_F = F_samples[min_index_original]
        optimal_n = n_samples[min_index_original]
        
        optimal_ns[i] = optimal_n
        min_F_values[i] = min_F
        
        # 使用精确导数计算偏导数
        try:
            dF_dr = exact_partial_F_r(optimal_n, r)
            partial_F_r_values[i] = dF_dr
        except Exception as e:
            # 如果计算失败，尝试使用数值导数
            try:
                # 使用中心差分法计算数值导数
                h = 1e-6
                F_plus = F(optimal_n, r + h)
                F_minus = F(optimal_n, r - h)
                if np.isfinite(F_plus) and np.isfinite(F_minus):
                    dF_dr = (F_plus - F_minus) / (2 * h)
                    partial_F_r_values[i] = dF_dr
                else:
                    partial_F_r_values[i] = np.nan
            except:
                partial_F_r_values[i] = np.nan
    
    # 过滤有效数据
    valid_mask = np.isfinite(optimal_ns) & np.isfinite(min_F_values)
    
    # 创建一个结果字典，包含所有重要结果
    result = {
        'L': L,
        'r_values': r_values[valid_mask],
        'optimal_ns': optimal_ns[valid_mask],
        'min_F_values': min_F_values[valid_mask],
        'force': partial_F_r_values[valid_mask]
    }
    
    return result


def save_results_column_format(results_list, base_dir):
    """
    按列存储结果，逐列写入文件，避免内存占用过大
    
    参数:
    results_list: 结果字典列表
    base_dir: 基础目录路径
    """
    # 确保目录存在
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. 保存L值到txt文件
    L_values = [result['L'] for result in results_list]
    L_filepath = os.path.join(base_dir, "L_values.txt")
    np.savetxt(L_filepath, L_values, fmt='%.6f')
    print(f"✅ L值已保存到 {L_filepath}")
    
    # 2. 创建CSV文件并逐列写入数据
    r_filepath = os.path.join(base_dir, "r_values.csv")
    f_filepath = os.path.join(base_dir, "f_values.csv")
    n_filepath = os.path.join(base_dir, "n_values.csv")
    
    # 记录每个样本的数据长度
    data_lengths = [len(result['r_values']) for result in results_list]
    max_length = max(data_lengths) if data_lengths else 0
    
    # 逐列写入r值
    print("📊 开始写入r值数据...")
    with open(r_filepath, 'w', newline='') as f:
        # 写入列名
        col_names = [f"sample_{i+1}" for i in range(len(results_list))]
        f.write(",".join(col_names) + "\n")
        
        # 逐行写入数据
        for row_idx in range(max_length):
            row_data = []
            for col_idx, result in enumerate(results_list):
                if row_idx < len(result['r_values']):
                    row_data.append(f"{result['r_values'][row_idx]:.10f}")
                else:
                    row_data.append("")  # 空值
            f.write(",".join(row_data) + "\n")
    
    print(f"✅ r值已逐列保存到 {r_filepath}")
    
    # 逐列写入f值
    print("📊 开始写入f值数据...")
    with open(f_filepath, 'w', newline='') as f:
        # 写入列名
        col_names = [f"sample_{i+1}" for i in range(len(results_list))]
        f.write(",".join(col_names) + "\n")
        
        # 逐行写入数据
        for row_idx in range(max_length):
            row_data = []
            for col_idx, result in enumerate(results_list):
                if row_idx < len(result['force']):
                    row_data.append(f"{result['force'][row_idx]:.10f}")
                else:
                    row_data.append("")  # 空值
            f.write(",".join(row_data) + "\n")
    
    print(f"✅ f值已逐列保存到 {f_filepath}")
    
    # 逐列写入n值
    print("📊 开始写入n值数据...")
    with open(n_filepath, 'w', newline='') as f:
        # 写入列名
        col_names = [f"sample_{i+1}" for i in range(len(results_list))]
        f.write(",".join(col_names) + "\n")
        
        # 逐行写入数据
        for row_idx in range(max_length):
            row_data = []
            for col_idx, result in enumerate(results_list):
                if row_idx < len(result['optimal_ns']):
                    row_data.append(f"{result['optimal_ns'][row_idx]:.10f}")
                else:
                    row_data.append("")  # 空值
            f.write(",".join(row_data) + "\n")
    
    print(f"✅ n值已逐列保存到 {n_filepath}")
    
    # 3. 保存统计数据
    stats_filepath = os.path.join(base_dir, "statistics.txt")
    with open(stats_filepath, 'w') as f:
        f.write(f"样本总数: {len(results_list)}\n")
        f.write(f"最大数据长度: {max_length}\n")
        f.write(f"L值范围: [{min(L_values):.2f}, {max(L_values):.2f}]\n")
        
        # 计算force的统计信息
        all_forces = []
        for result in results_list:
            if 'force' in result:
                all_forces.extend(result['force'])
        
        if all_forces:
            all_forces = np.array(all_forces)
            f.write(f"Force统计信息:\n")
            f.write(f"  最小值: {np.nanmin(all_forces):.6f}\n")
            f.write(f"  最大值: {np.nanmax(all_forces):.6f}\n")
            f.write(f"  平均值: {np.nanmean(all_forces):.6f}\n")
            f.write(f"  标准差: {np.nanstd(all_forces):.6f}\n")
    
    print(f"✅ 统计信息已保存到 {stats_filepath}")
    
    return L_filepath, r_filepath, f_filepath


def main():
    # 参数设置
    xi_f = 2.0
    U0 = 10.0
    xi_u = 30.0
    Num = 100 # 运行300次

    # 取样参数
    mu = 400.0
    sigma = 150.0

    # 存储目录
    base_dir = "/home/tyt/project/Single-chain/opt+R/Rand_length/column_format"

    # energy values
    data = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]

    all_results = []
    valid_count = 0

    print("🚀 开始批量优化计算...")
    
    for i in range(Num):
        if (i+1) % 10 == 0:
            print(f'  优化计算进度: {i + 1}/{Num}')

        L = np.random.normal(mu, sigma)
        
        # 检查L是否合理
        min_L = 10 * xi_u
        max_L = 10 * xi_u + 200.0
        if L <= min_L or L >= max_L:
            print(f"  L = {L:.2f} 超出范围，跳过")
            continue
        else:
            print(f"  计算 L = {L:.2f}")
            try:
                result = optimization_function(U0, L, xi_u, xi_f, data)
                
                # 检查结果是否有效
                if len(result['r_values']) > 10:  # 至少有10个有效点
                    # 收集结果
                    all_results.append(result)
                    valid_count += 1
                    print(f"    成功计算，有效数据点: {len(result['r_values'])}")
                else:
                    print(f"    数据点过少，跳过")
            except Exception as e:
                print(f"    计算失败: {e}")
                continue
    
    # 保存结果
    if all_results:
        print(f"\n开始保存结果，有效样本数: {valid_count}")
        save_results_column_format(all_results, base_dir)
    else:
        print("⚠️ 没有有效样本，无法保存结果")
    
    print(f'\n🎉 处理完成. 有效样本数: {valid_count}/{Num}')
    
    return 0


if __name__ == "__main__":
    main()
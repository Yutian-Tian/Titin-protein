import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import os

def calculate_pf(E, f, rs):
    """计算 p_f"""
    return 1 / (1 + np.exp(-E + f * rs))

def calculate_f(r, Lc):
    """计算 f"""
    if Lc <= 0 or r >= Lc:
        return float('nan')
    ratio = r / Lc
    return 0.25 * ((1 - ratio)**(-2) - 1 + 4 * ratio)

def equations(vars, r, params):
    """定义需要求解的方程组"""
    f, Lc = vars
    L, E, rs, N, xi = params
    
    # 方程1: Lc 的定义
    pf = 1 / (1 + np.exp(-E + f * rs))
    eq1 = Lc - (L - pf * N * xi)
    
    # 方程2: f 的定义
    if Lc <= 0 or r >= Lc:
        eq2 = float('nan')
    else:
        ratio = r / Lc
        eq2 = f - 0.25 * ((1 - ratio)**(-2) - 1 + 4 * ratio)
    
    return [eq1, eq2]

def solve_for_r(r, params, initial_guess=None):
    """对于给定的r，解方程组求f和Lc"""
    if initial_guess is None:
        initial_guess = [0.1, params[0] * 0.9]  # 初始猜测: [f, Lc]
    
    try:
        solution = fsolve(equations, initial_guess, args=(r, params), full_output=True)
        f, Lc = solution[0]
        
        # 检查解的有效性
        if solution[2] != 1:  # ier != 1 表示求解失败
            return float('nan'), float('nan')
        
        # 验证物理意义
        if Lc <= 0 or f < 0 or r >= Lc:
            return float('nan'), float('nan')
            
        return f, Lc
    except:
        return float('nan'), float('nan')

def iterative_calculation(r_values, initial_params):
    """改进的迭代计算：对于每个r独立求解方程组"""
    # 初始化结果数组
    f_values = np.zeros(len(r_values))
    Lc_values = np.zeros(len(r_values))
    pf_values = np.zeros(len(r_values))  # 新增：保存pf值
    
    # 准备参数元组
    params = (initial_params['L'], initial_params['E'], 
              initial_params['rs'], initial_params['N'], 
              initial_params['xi'])
    
    # 初始猜测
    prev_f, prev_Lc = 0.1, initial_params['L'] * 0.9
    
    for i, r in enumerate(r_values):
        # 使用前一个解作为初始猜测（连续性假设）
        initial_guess = [prev_f, prev_Lc]
        
        # 解方程组
        f_current, Lc_current = solve_for_r(r, params, initial_guess)
        
        # 如果求解失败，尝试其他初始猜测
        if np.isnan(f_current) or np.isnan(Lc_current):
            # 尝试不同的初始猜测
            for guess_f in [0.01, 0.1, 1.0, 10.0]:
                for guess_Lc in [initial_params['L']*0.5, initial_params['L']*0.8, initial_params['L']*0.95]:
                    f_current, Lc_current = solve_for_r(r, params, [guess_f, guess_Lc])
                    if not (np.isnan(f_current) or np.isnan(Lc_current)):
                        break
                if not (np.isnan(f_current) or np.isnan(Lc_current)):
                    break
        
        # 存储结果
        f_values[i] = f_current
        Lc_values[i] = Lc_current
        
        # 计算并存储pf值
        if not (np.isnan(f_current) or np.isnan(Lc_current)):
            pf_values[i] = calculate_pf(initial_params['E'], f_current, initial_params['rs'])
        else:
            pf_values[i] = float('nan')
        
        # 更新前一个解（如果当前解有效）
        if not (np.isnan(f_current) or np.isnan(Lc_current)):
            prev_f, prev_Lc = f_current, Lc_current
        else:
            # 如果当前解无效，将后续值设为nan
            f_values[i:] = np.nan
            Lc_values[i:] = np.nan
            pf_values[i:] = np.nan
            print(f"在 r={r:.2f} 之后无有效解")
            break
    
    return f_values, Lc_values, pf_values

def save_results(r_values, f_values, Lc_values, pf_values, initial_params, filename=None):
    """保存计算结果到CSV文件"""
    if filename is None:
        # 根据参数自动生成文件名
        filename = f"/home/tyt/project/Single-chain/Average_theory/iteration_results.csv"
    
    # 创建有效数据掩码
    valid_mask = ~np.isnan(f_values)
    
    # 准备数据
    data_to_save = np.column_stack((
        r_values[valid_mask],
        f_values[valid_mask],
        Lc_values[valid_mask],
        pf_values[valid_mask]
    ))
    
    # 保存到文件
    header = f"# Parameters: L={initial_params['L']}, E={initial_params['E']}, rs={initial_params['rs']}, N={initial_params['N']}, xi={initial_params['xi']}\n"
    header += "r,f,Lc,pf"
    
    np.savetxt(filename, data_to_save, delimiter=',', header=header, comments='')
    
    # 保存参数到单独的文件
    param_filename = os.path.splitext(filename)[0] + "_params.txt"
    with open(param_filename, 'w') as f:
        f.write("Simulation Parameters:\n")
        f.write("=" * 40 + "\n")
        for key, value in initial_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write(f"Data points: {np.sum(valid_mask)}\n")
        f.write(f"r range: [{r_values[0]:.4f}, {r_values[np.where(valid_mask)[0][-1]]:.4f}]\n")
    
    print(f"结果已保存到: {filename}")
    print(f"参数文件已保存到: {param_filename}")
    
    return filename
    

def print_summary(r_values, f_values, Lc_values, pf_values, initial_params):
    """打印计算摘要"""
    valid_mask = ~np.isnan(f_values)
    valid_count = np.sum(valid_mask)
    
    print("\n" + "=" * 60)
    print("计算结果摘要")
    print("=" * 60)
    print(f"总数据点: {len(r_values)}")
    print(f"有效数据点: {valid_count}")
    print(f"无效数据点: {len(r_values) - valid_count}")
    
    if valid_count > 0:
        valid_indices = np.where(valid_mask)[0]
        r_min = r_values[valid_indices[0]]
        r_max = r_values[valid_indices[-1]]
        print(f"有效r范围: [{r_min:.4f}, {r_max:.4f}]")
        
        # 查找f的最大值及其对应的r
        max_f_idx = np.nanargmax(f_values)
        max_f = f_values[max_f_idx]
        max_f_r = r_values[max_f_idx]
        print(f"最大f值: {max_f:.6f} (在 r={max_f_r:.4f} 处)")
        
        # 查找Lc的最小值
        min_Lc_idx = np.nanargmin(Lc_values)
        min_Lc = Lc_values[min_Lc_idx]
        min_Lc_r = r_values[min_Lc_idx]
        print(f"最小Lc值: {min_Lc:.6f} (在 r={min_Lc_r:.4f} 处)")
        
        # 打印头尾几个点
        print("\n关键点结果:")
        if valid_count <= 10:
            indices = valid_indices
        else:
            indices = np.concatenate([
                valid_indices[:3],
                valid_indices[valid_count//2 - 1:valid_count//2 + 2],
                valid_indices[-3:]
            ])
        
        for idx in indices:
            r = r_values[idx]
            f = f_values[idx]
            Lc = Lc_values[idx]
            pf = pf_values[idx]
            print(f"r={r:8.4f}: f={f:10.6f}, Lc={Lc:10.4f}, pf={pf:8.6f}")

def main():
    # 初始化参数
    initial_params = {
        'L': 350.0,      # 初始长度
        'E': 3.0,       # 能量参数
        'rs': 30.0,       # rs参数
        'N': 10.0,       # N值
        'xi': 30.0       # ξ值
    }
    
    print("初始参数:")
    for key, value in initial_params.items():
        print(f"  {key}: {value}")
    
    # 创建r的取值范围
    # 根据物理意义，r应小于L
    max_r = initial_params['L'] * 0.99
    r_values = np.linspace(0, max_r, 1000)
    
    print(f"\n计算范围: r ∈ [0, {max_r:.2f}]")
    print(f"数据点数量: {len(r_values)}")
    print("进行求解...")
    
    # 执行计算
    f_values, Lc_values, pf_values = iterative_calculation(r_values, initial_params)
    
    # 打印摘要
    print_summary(r_values, f_values, Lc_values, pf_values, initial_params)
    
    # 自动保存结果
    csv_filename = save_results(r_values, f_values, Lc_values, pf_values, initial_params)
    
    # 提供进一步分析选项
    print("\n" + "=" * 60)
    print("计算完成!")
    print("=" * 60)
    print(f"1. 结果已保存到: {csv_filename}")
    

if __name__ == "__main__":
    main()
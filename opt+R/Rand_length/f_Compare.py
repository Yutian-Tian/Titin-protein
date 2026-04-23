# 比较模拟与理论的结果

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import fsolve
import os
import warnings

# ==================== 第一个程序的核心函数 ====================
def process_and_plot_first_program(f_limit, ffile_path, rfile_path):
    """第一个程序的处理函数，返回处理后的数据"""
    
    # 1. 读取数据文件
    try:
        f_data = pd.read_csv(ffile_path)
        r_data = pd.read_csv(rfile_path)
    except FileNotFoundError as e:
        print(f"文件读取错误: {e}")
        return None
    
    # 确保两个文件有相同的列数
    if f_data.shape[1] != r_data.shape[1]:
        print("错误: 两个文件的列数不一致!")
        return None
    
    # 获取列名
    columns = f_data.columns
    num_groups = len(columns)
    
    print(f"找到 {num_groups} 组数据")
    
    # 2. 数据处理和插值准备
    processed_groups = []
    all_r_values = []
    
    for col in columns:
        # 获取当前组的f和r数据
        f_col = f_data[col].values
        r_col = r_data[col].values
        
        # 移除NaN值
        valid_mask = ~np.isnan(f_col) & ~np.isnan(r_col)
        f_valid = f_col[valid_mask]
        r_valid = r_col[valid_mask]
        
        if len(f_valid) < 2:
            continue
        
        # 添加到所有r值集合中
        all_r_values.extend(r_valid.tolist())
        
        processed_groups.append({
            'name': col,
            'f': f_valid,
            'r': r_valid
        })
    
    if not processed_groups:
        print("错误: 没有可处理的数据!")
        return None
    
    # 获取所有组的r值并集，并排序去重
    all_r_unique = np.sort(np.unique(all_r_values))
    print(f"所有组r值的并集范围: [{all_r_unique[0]:.4f}, {all_r_unique[-1]:.4f}], 共 {len(all_r_unique)} 个点")
    
    # 3. 分段线性插值
    interpolated_f_values = []
    valid_groups = []
    
    for group in processed_groups:
        f_vals = group['f']
        r_vals = group['r']
        
        # 确保r值是递增的（插值需要）
        sort_idx = np.argsort(r_vals)
        r_sorted = r_vals[sort_idx]
        f_sorted = f_vals[sort_idx]
        
        # 去除重复的r值
        r_unique, idx_unique = np.unique(r_sorted, return_index=True)
        f_unique = f_sorted[idx_unique]
        
        if len(r_unique) < 2:
            continue
        
        # 创建插值函数
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f_interp_func = interpolate.interp1d(
                    r_unique, 
                    f_unique, 
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
            
            # 在r并集范围内插值
            f_interp = f_interp_func(all_r_unique)
            interpolated_f_values.append(f_interp)
            valid_groups.append(group)
            
        except Exception as e:
            continue
    
    if not interpolated_f_values:
        print("错误: 没有成功插值的数据!")
        return None
    
    # 转换为数组
    interpolated_f_array = np.array(interpolated_f_values)
    
    # 4. 计算平均值fa（忽略NaN值）
    fa = np.nanmean(interpolated_f_array, axis=0)
    
    # 5. 寻找最小的临界点rc（第一个fa>=f_limit的位置）
    rc = None
    valid_mask = None
    
    # 遍历所有点，找到第一个fa>=f_limit的位置
    for i in range(len(fa)):
        if fa[i] >= f_limit:
            rc = all_r_unique[i]
            # 取临界点之前的所有点（包括刚好到达临界点的点）
            valid_mask = np.arange(len(fa)) <= i
            break
    
    # 如果没有找到临界点（所有fa都小于f_limit）
    if rc is None:
        rc = all_r_unique[-1]
        valid_mask = np.ones(len(fa), dtype=bool)
    
    # 获取有效的r和fa
    r_valid = all_r_unique[valid_mask]
    fa_valid = fa[valid_mask]
    
    print(f"第一个程序 - 最小临界点 rc = {rc:.4f}")
    print(f"第一个程序 - 有效数据范围: [{r_valid[0]:.4f}, {r_valid[-1]:.4f}], 共 {len(r_valid)} 个点")
    
    return {
        'r_common': all_r_unique,
        'fa': fa,
        'r_valid': r_valid,
        'fa_valid': fa_valid,
        'rc': rc,
        'num_groups': len(valid_groups),
        'valid_groups': valid_groups
    }

# ==================== 第二个程序的核心函数 ====================
def calculate_pf(E, f, rs):
    """计算 p_f"""
    if f*rs >= 1e14:
        return 0.0
    else:
        return 1 / (1 + np.exp(-E + f * rs))

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
#        eq2 = f - 0.25 * ((1 - ratio)**(-2) - 1 + 4 * ratio)
        eq2 = f + np.pi**2 * ratio / Lc**2 - 4 * ratio / (np.pi * (1 - ratio**2)) 
    
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
        
        # 更新前一个解（如果当前解有效）
        if not (np.isnan(f_current) or np.isnan(Lc_current)):
            prev_f, prev_Lc = f_current, Lc_current
        else:
            # 如果当前解无效，将后续值设为nan
            f_values[i:] = np.nan
            Lc_values[i:] = np.nan
            print(f"第二个程序 - 在 r={r:.2f} 之后无有效解")
            break
    
    return f_values, Lc_values

def process_second_program(initial_params, f_limit, rc_first):
    """第二个程序的处理函数，返回处理后的数据"""
    print("\n第二个程序参数:")
    for key, value in initial_params.items():
        print(f"  {key}: {value}")
    
    # 创建r的取值范围 - 限制在第一个程序的rc范围内
    max_r = min(initial_params['L'] * 0.99, rc_first * 1.2)  # 稍微超出rc一点
    r_values = np.linspace(0, max_r, 1000)
    
    print(f"\n第二个程序计算范围: r ∈ [0, {max_r:.2f}]")
    print(f"数据点数量: {len(r_values)}")
    print("进行求解...")
    
    # 执行计算
    f_values, Lc_values = iterative_calculation(r_values, initial_params)
    
    # 找到第一个f >= f_limit的位置
    valid_mask = ~np.isnan(f_values)
    if np.any(valid_mask):
        for i in range(len(f_values)):
            if valid_mask[i] and f_values[i] >= f_limit:
                # 截断到第一个超过f_limit的点
                f_values = f_values[:i+1]
                Lc_values = Lc_values[:i+1]
                r_values = r_values[:i+1]
                print(f"第二个程序 - 在 r={r_values[i]:.2f} 处 f={f_values[i]:.4f} >= f_limit={f_limit}")
                break
    
    return {
        'r_values': r_values,
        'f_values': f_values,
        'Lc_values': Lc_values,
        'valid_mask': ~np.isnan(f_values)
    }

# ==================== 主绘图函数 ====================
def plot_combined_results(first_data, second_data, f_limit, save_path):
    """将两个程序的结果绘制在同一张图中"""
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 获取第一个程序的数据
    r_first = first_data['r_valid']
    fa_first = first_data['fa_valid']
    rc_first = first_data['rc']
    num_groups = first_data['num_groups']
    valid_groups = first_data['valid_groups']
    
    # 获取第二个程序的数据
    r_second = second_data['r_values']
    f_second = second_data['f_values']
    valid_mask_second = second_data['valid_mask']
    
    # 只绘制第二个程序的有效数据
    if np.any(valid_mask_second):
        r_second_valid = r_second[valid_mask_second]
        f_second_valid = f_second[valid_mask_second]
    else:
        r_second_valid = np.array([])
        f_second_valid = np.array([])
    
    # 1. 绘制第一个程序的背景数据（灰色线）
    print("绘制第一个程序的背景数据...")
    plotted_bg = False  # 标记是否已经绘制过背景数据
    for i, group in enumerate(valid_groups):
        if i < 100:  # 限制背景线数量，避免过于密集
            # 只绘制r <= rc_first的数据
            mask = (group['r'] <= rc_first) & (group['f'] <= f_limit * 1.2)
            if np.sum(mask) > 1:  # 至少2个点才能画线
                plt.plot(group['r'][mask], group['f'][mask], 
                        color='gray', 
                        alpha=0.15, 
                        linewidth=0.5,
                        label='Experimental Data (individual)' if not plotted_bg else '_nolegend_')
                plotted_bg = True
    
    # 2. 绘制第一个程序的平均线（红色线）
    if len(r_first) > 0:
        plt.plot(r_first, fa_first, 
                color='red', 
                linewidth=3, 
                label=f'Experimental Average (n={num_groups})')
    
    # 3. 绘制第二个程序的理论曲线（蓝色线）
    if len(r_second_valid) > 1:
        plt.plot(r_second_valid, f_second_valid, 
                color='blue', 
                linewidth=3, 
                linestyle='--',
                label='Theoretical Model')
    
    # 4. 标记临界点和f_limit
    if rc_first > 0:
        plt.axvline(x=rc_first, color='red', linestyle=':', 
                    linewidth=2, alpha=0.7, label=f'Experimental $r_c$ = {rc_first:.2f}')
    
    plt.axhline(y=f_limit, color='green', linestyle=':', 
                linewidth=2, alpha=0.7, label=f'$f$ = {f_limit}')
    
    # 5. 设置图形属性
    plt.xlabel('End-to-end Distance $r$', fontsize=14, fontweight='bold')
    plt.ylabel('Force $f$', fontsize=14, fontweight='bold')
    plt.title(f'Force-Distance Relationship: Experimental vs Theoretical ($f < {f_limit}$)', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    
    # 6. 设置坐标轴范围
    # x轴范围：0到最大有效值
    x_max = 0
    if len(r_first) > 0:
        x_max = max(x_max, np.max(r_first))
    if len(r_second_valid) > 0:
        x_max = max(x_max, np.max(r_second_valid))
    
    if x_max > 0:
        x_max = min(x_max * 1.05, rc_first * 1.5 if rc_first > 0 else x_max * 1.1)
        plt.xlim(0, x_max)
    
    # y轴范围：0到f_limit加上一些边距
    y_max = f_limit * 1.2
    plt.ylim(0, y_max)
    
    plt.tight_layout()
    
    # 7. 保存图片
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n合并图形已保存到: {save_path}")
    
    
    # 返回统计信息
    stats = {
        'rc_first': rc_first,
        'num_groups': num_groups,
        'r_first_range': [r_first[0], r_first[-1]] if len(r_first) > 0 else [0, 0],
        'r_second_range': [r_second_valid[0], r_second_valid[-1]] if len(r_second_valid) > 0 else [0, 0]
    }
    
    return stats

def main():
    print("=== 合并两个程序的结果 - 实验数据 vs 理论模型 ===\n")
    
    # ==================== 参数设置 ====================
    # 第一个程序的参数
    f_limit = 15.0
    ffile_path = '/home/tyt/project/Single-chain/opt+R/Rand_length/column_format/f_values.csv'
    rfile_path = '/home/tyt/project/Single-chain/opt+R/Rand_length/column_format/r_values.csv'
    
    # 第二个程序的参数
    initial_params = {
        'L': 350.0,      # 初始长度
        'E': 3.0,       # 能量参数
        'rs': 30.0,     # rs参数
        'N': 10.0,      # N值
        'xi': 30.0      # ξ值
    }
    
    # 保存路径
    save_path = "/home/tyt/project/Single-chain/combined_f_r_curves.png"
    
    # ==================== 运行第一个程序 ====================
    print("="*60)
    print("运行第一个程序（实验数据处理）...")
    print("="*60)
    
    first_data = process_and_plot_first_program(f_limit, ffile_path, rfile_path)
    
    if first_data is None:
        print("第一个程序处理失败!")
        return
    
    print(f"\n第一个程序完成!")
    print(f"找到临界点 rc = {first_data['rc']:.4f}")
    print(f"有效数据点: {len(first_data['r_valid'])}")
    
    # ==================== 运行第二个程序 ====================
    print("\n" + "="*60)
    print("运行第二个程序（理论模型计算）...")
    print("="*60)
    
    second_data = process_second_program(initial_params, f_limit, first_data['rc'])
    
    if np.any(second_data['valid_mask']):
        valid_count = np.sum(second_data['valid_mask'])
        print(f"\n第二个程序完成!")
        print(f"有效数据点: {valid_count}")
        
        # 找到最大f值
        f_max_idx = np.nanargmax(second_data['f_values'])
        f_max = second_data['f_values'][f_max_idx]
        r_at_f_max = second_data['r_values'][f_max_idx]
        print(f"理论模型最大f值: {f_max:.4f} (在 r={r_at_f_max:.2f} 处)")
    else:
        print("第二个程序没有有效数据!")
    
    # ==================== 合并绘图 ====================
    print("\n" + "="*60)
    print("合并绘图...")
    print("="*60)
    
    stats = plot_combined_results(first_data, second_data, f_limit, save_path)
    
    # ==================== 打印统计摘要 ====================
    print("\n" + "="*60)
    print("统计摘要")
    print("="*60)
    print(f"实验数据:")
    print(f"  - 数据组数: {stats['num_groups']}")
    print(f"  - 临界点 rc: {stats['rc_first']:.4f}")
    if stats['r_first_range'][1] > 0:
        print(f"  - r范围: [{stats['r_first_range'][0]:.2f}, {stats['r_first_range'][1]:.2f}]")
    
    print(f"\n理论模型:")
    print(f"  - 参数: L={initial_params['L']}, E={initial_params['E']}, rs={initial_params['rs']}, "
          f"N={initial_params['N']}, ξ={initial_params['xi']}")
    if stats['r_second_range'][1] > 0:
        print(f"  - r范围: [{stats['r_second_range'][0]:.2f}, {stats['r_second_range'][1]:.2f}]")
    
    if save_path:
        print(f"\n合并图形已保存到: {save_path}")

if __name__ == "__main__":
    main()
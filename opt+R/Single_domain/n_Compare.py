"""
用于比较定力系综的数值结果与2-state理论
比对数值模拟与2-state理论
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.optimize import fsolve

# ========== 参数设置 ==========
# 归一化参数N（domain全部打开时n=1）
N = 1.0

# ========== 程序一的部分：读取CSV数据并处理 ==========
def read_csv_files(file1_path, file2_path):
    """
    读取两个CSV文件
    """
    try:
        # 读取CSV文件，不使用表头，从第一行开始读取
        df1 = pd.read_csv(file1_path, header=1)
        df2 = pd.read_csv(file2_path, header=1)
        
        return df1.values, df2.values
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None
    except Exception as e:
        print(f"Error reading files: {e}")
        return None, None

def process_experimental_data(r_data, n_data, N=1.0):
    """
    处理实验数据，将n除以N归一化，并计算平均值
    对于超出数据范围的部分：
      - 小于数据范围[a,b]的部分，插值为0
      - 大于数据范围[a,b]的部分，插值为N
    
    返回:
    r_uniform: 统一的r值网格
    normalized_avg: 归一化后的平均值曲线
    all_normalized_curves: 所有归一化后的原始曲线
    """
    if r_data is None or n_data is None:
        print("Data arrays are empty. Cannot process.")
        return None, None, None
    
    # 获取列数，确保两个文件列数相同
    if len(r_data.shape) == 2:
        num_trajectories = min(r_data.shape[1], n_data.shape[1])
        print(f"读取到 {num_trajectories} 条轨迹数据")
    else:
        print(f"错误: 数据不是二维数组")
        return None, None, None
    
    # 用于平均值计算的数据（所有原始数据）
    all_curves_data = []
    
    # 收集所有r值范围，用于生成统一的插值r数组
    all_r_min = []
    all_r_max = []
    
    # 首先处理所有数据，准备用于平均值计算
    for i in range(num_trajectories):
        r = r_data[:, i]
        n = n_data[:, i]
        
        # 移除NaN值
        valid_mask = ~np.isnan(r) & ~np.isnan(n)
        r_valid = r[valid_mask]
        n_valid = n[valid_mask]
        
        # 保存所有原始数据
        all_curves_data.append((r_valid.copy(), n_valid.copy()))
        
        # 收集r值范围
        if len(r_valid) > 0:
            all_r_min.append(np.min(r_valid))
            all_r_max.append(np.max(r_valid))
    
    if len(all_curves_data) == 0:
        print("No valid experimental data found.")
        return None, None, None
    
    # ========== 平均值计算部分 ==========
    # 确定用于平均值计算的r值范围
    r_min = min(all_r_min)  # 所有轨迹的最小r值
    r_max = max(all_r_max)  # 所有轨迹的最大r值
    
    print(f"轨迹r范围统计:")
    print(f"  所有轨迹最小r值: {r_min:.3f} (a)")
    print(f"  所有轨迹最大r值: {r_max:.3f} (b)")
    print(f"  单个轨迹平均r范围: [{np.mean(all_r_min):.3f}, {np.mean(all_r_max):.3f}]")
    
    # 生成用于平均值计算的统一r值数组
    # 扩展范围以便查看边界效应
    r_uniform_min = max(0, r_min * 0.5)  # 确保非负
    r_uniform_max = r_max * 1.1  # 扩展10%以便查看边界
    r_uniform = np.linspace(r_uniform_min, r_uniform_max, 1000)
    nsum = np.zeros_like(r_uniform)
    count = np.zeros_like(r_uniform)
    
    print(f"统一网格: 从{r_uniform_min:.3f}到{r_uniform_max:.3f}, 共{len(r_uniform)}个点")
    
    # 对所有数据进行插值并求和
    for idx, (r_orig, n_orig) in enumerate(all_curves_data):
        # 对每条轨迹，获取其实际数据范围[a_i, b_i]
        a_i = np.min(r_orig)
        b_i = np.max(r_orig)
        
        # 线性插值，对于超出原始数据范围的点:
        # - 小于a_i的部分填充0
        # - 大于b_i的部分填充N
        interp_func = interpolate.interp1d(
            r_orig, n_orig, kind='linear', 
            bounds_error=False, 
            fill_value=(0.0, N)  # 元组，第一个元素用于小于范围，第二个元素用于大于范围
        )
        
        # 在统一r值下计算插值n值
        n_interp = interp_func(r_uniform)
        
        # 累加所有值
        nsum += n_interp
        count += 1
        
        # 显示前几条轨迹的详细信息
        if idx < 3:
            below_range = np.sum(r_uniform < a_i)
            above_range = np.sum(r_uniform > b_i)
            in_range = np.sum((r_uniform >= a_i) & (r_uniform <= b_i))
            print(f"  轨迹 {idx+1}: r范围=[{a_i:.3f}, {b_i:.3f}], "
                  f"小于范围: {below_range}个点填充0, "
                  f"大于范围: {above_range}个点填充N={N}, "
                  f"范围内: {in_range}个点")
    
    # 计算平均值（所有点都参与计算）
    n_avg = nsum / count
    
    # 归一化：除以N
    normalized_avg = n_avg / N
    
    # 收集所有归一化后的原始曲线（用于绘制灰色半透明线）
    all_normalized_curves = []
    for r_orig, n_orig in all_curves_data:
        normalized_n = n_orig / N
        all_normalized_curves.append((r_orig.copy(), normalized_n.copy()))
    
    print(f"\n插值填充策略:")
    print(f"  小于数据范围[a,b]的部分: 填充0")
    print(f"  大于数据范围[a,b]的部分: 填充N={N}")
    print(f"  数据范围内[a,b]: 线性插值")
    
    return r_uniform, normalized_avg, all_normalized_curves

# ========== 程序二的部分：理论计算（使用fsolve） ==========
def p_f(f, E, xi_f, xi_u):
    """数值稳定的折叠概率计算"""
    # 原始公式: p_f(f) = 1 / (1 + exp(-E + f*(ξ_u-ξ_f)))
    term = -E + f * (xi_u - xi_f)
    
    # 使用数值稳定的sigmoid函数
    # 当term很大时，使用近似值
    if term > 700.0:  # 使用更大的阈值，因为E可能很大
        return 0.0
    elif term < -700.0:
        return 1.0
    else:
        return 1.0 / (1.0 + np.exp(term))

def equation_to_solve(f, r, E, xi_f, xi_u):
    """定义需要求解的方程"""
    # 计算折叠概率
    pf = p_f(f, E, xi_f, xi_u)
    
    # 计算轮廓长度
    Lc = xi_u - pf * (xi_u - xi_f)
    
    # 检查轮廓长度是否有效
    if Lc <= 0:
        # 返回大值，使求解器远离无效区域
        return 1e10
    
    # 计算端端比
    x = r / Lc
    
    # 检查x是否在有效范围内
    if x >= 1 or x <= -1:
        # 返回大值，使求解器远离无效区域
        return 1e10
    
    # 力方程：f - rhs = 0
    rhs = - (np.pi**2 * x) / (Lc**2) + (4 * x) / (np.pi * (1 - x**2)**2)
    
    return f - rhs

def calculate_theoretical_curves_fsolve(E, xi_f, xi_u, num_points=1000, maxfev=10000):
    """
    使用fsolve计算理论曲线
    求解范围固定为 [0, 0.95*xi_u]
    
    参数:
    E: 能量参数
    xi_f: 折叠状态长度
    xi_u: 展开状态长度
    num_points: r值点数
    maxfev: 最大函数调用次数
    """
    
    # 生成r值数组，固定范围 [0, 0.95*xi_u]
    r_max = 0.95 * xi_u
    r_values = np.linspace(0, r_max, num_points)
    
    # 存储结果的数组
    f_values = np.zeros_like(r_values)  # 力f
    pf_values = np.zeros_like(r_values) # 折叠概率p_f
    puf_values = np.zeros_like(r_values) # 展开概率p_u = 1 - p_f
    
    # 初始猜测值
    f_guess = 0.0
    
    print(f"开始使用fsolve求解理论曲线，固定范围: [0, {r_max:.3f}] (0.95*ξ_u)")
    
    for i, r in enumerate(r_values):
        # 跳过r=0的情况，避免除以零
        if r < 1e-10:
            f_values[i] = 0.0
            pf_values[i] = p_f(0.0, E, xi_f, xi_u)
            puf_values[i] = 1.0 - pf_values[i]
            continue
            
        # 使用前一个解作为初始猜测（连续性假设）
        if i > 0 and np.isfinite(f_values[i-1]):
            f_guess = f_values[i-1]
        else:
            f_guess = 0.0
        
        # 使用fsolve求解
        try:
            # 设置求解器参数
            # 使用fprime参数可以提高求解效率，但这里我们使用默认的有限差分
            f_solution = fsolve(
                equation_to_solve, 
                f_guess, 
                args=(r, E, xi_f, xi_u), 
                maxfev=maxfev,
                full_output=False
            )
            
            # 检查解是否合理
            if not np.isfinite(f_solution[0]):
                f_values[i] = np.nan
                pf_values[i] = np.nan
                puf_values[i] = np.nan
                continue
            
            f_val = f_solution[0]
            
            # 计算对应的折叠概率
            pf_val = p_f(f_val, E, xi_f, xi_u)
            p_u_val = 1.0 - pf_val
            
            # 检查解是否在物理合理范围内
            if abs(f_val) > 1000 or not np.isfinite(pf_val) or not np.isfinite(p_u_val):
                f_values[i] = np.nan
                pf_values[i] = np.nan
                puf_values[i] = np.nan
            else:
                f_values[i] = f_val
                pf_values[i] = pf_val
                puf_values[i] = p_u_val
                
        except Exception as e:
            # 求解失败，设置为NaN
            f_values[i] = np.nan
            pf_values[i] = np.nan
            puf_values[i] = np.nan
        
        # 显示进度
        if (i+1) % (num_points//10) == 0:
            print(f"  进度: {i+1}/{num_points} ({(i+1)/num_points*100:.0f}%)")
    
    # 理论解不进行插值，直接返回原始数据（可能包含NaN）
    
    # 统计有效点
    valid_count = np.sum(np.isfinite(puf_values))
    print(f"理论曲线求解完成，有效点: {valid_count}/{len(r_values)}")
    
    # 理论解不进行插值，直接返回原始数据（可能包含NaN）
    return r_values, f_values, pf_values, puf_values

# ========== 整合可视化部分 ==========
def visualize_combined_curves(exp_r, exp_avg, exp_curves, theory_r, theory_puf, xi_u, params, savepath):
    """
    在同一张图中绘制实验数据（归一化后）和理论曲线
    
    参数:
    exp_r: 实验数据统一的r值网格
    exp_avg: 归一化后的实验平均值
    exp_curves: 所有归一化后的实验原始曲线
    theory_r: 理论计算的r值（范围：[0, 0.95*xi_u]）
    theory_puf: 理论计算的p_uf值（可能包含NaN）
    xi_u: 展开状态长度，用于确定理论曲线范围
    params: 参数字典
    savepath: 保存路径
    """
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # ========== 绘制实验数据 ==========
    # 1. 绘制所有原始实验曲线（灰色半透明）
    GRAY_COLOR = 'gray'
    GRAY_ALPHA = 0.15
    GRAY_LINEWIDTH = 1.0
    
    # 只绘制前50条轨迹，避免图形过于拥挤
    num_trajectories_to_plot = len(exp_curves)
    
    for i, (r_orig, n_orig) in enumerate(exp_curves[:num_trajectories_to_plot]):
        if len(r_orig) > 2:  # 至少需要3个点才能绘制曲线
            # 线性插值
            interp_func = interpolate.interp1d(
                r_orig, n_orig, kind='linear', 
                bounds_error=False, fill_value=(0.0, N)
            )
            
            # 生成该曲线的r值网格
            r_curve = np.linspace(np.min(r_orig), np.max(r_orig), 100)
            n_interp = interp_func(r_curve)
            
            # 绘制插值曲线（半透明灰线）
            plt.plot(r_curve, n_interp, 
                    color=GRAY_COLOR, linewidth=GRAY_LINEWIDTH, alpha=GRAY_ALPHA,
                    zorder=1)
    
    # 2. 绘制实验平均值曲线（蓝色）
    plt.plot(exp_r, exp_avg, 
            color='blue', linewidth=3.0, linestyle='-',
            label=f'Numerical Simulation Average (n={len(exp_curves)})', 
            alpha=0.9, zorder=3)
    
    # ========== 绘制理论曲线 ==========
    # 理论解不进行插值处理，绘制时跳过NaN值
    # matplotlib的plot函数会自动跳过NaN值，但会在NaN处断开连线
    valid_theory_mask = ~np.isnan(theory_puf)
    
    if np.any(valid_theory_mask):
        # 绘制理论曲线，NaN处会自动断开
        plt.plot(theory_r, theory_puf, 
                color='green', linewidth=3.0, linestyle='--',
                label=f'Theoretical $p_u (r)$ (Two-State Model)]', 
                alpha=0.9, zorder=4)
        
        print(f"理论曲线绘制: 有效点 {np.sum(valid_theory_mask)}/{len(theory_r)}")
    else:
        print("警告: 理论曲线没有有效点，无法绘制")
    
    # ========== 设置图形属性 ==========
    plt.xlabel('End-to-end distance $r$', fontsize=16)
    plt.ylabel('Unfolded probability $p_u$', fontsize=16)
    
    # 设置标题，包含参数信息
    title = f'Numerical Simulation vs Two-State Model\n'
    plt.title(title, fontsize=20, pad=10)
    
    # 添加图例
    plt.legend(fontsize=12, loc='best')
    
    # 设置网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 设置刻度
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=14)
    
    # 设置坐标轴范围
    # 理论曲线固定范围 [0, 0.95*xi_u]
    theory_r_max = 0.95 * xi_u
    
    # 确定实验数据的有效范围
    exp_r_min = np.min(exp_r)
    exp_r_max = np.max(exp_r)
    
    # x轴范围：取理论范围和实验范围的并集，但要确保从0开始
    x_min = 0  # 总是从0开始
    x_max = max(exp_r_max, theory_r_max)  # 取最大值
    
    # 自动调整x轴范围，留5%边距
    x_margin = 0.05 * (x_max - x_min)
    plt.xlim(x_min - x_margin, x_max + x_margin)
    
    # 标记理论曲线范围
    plt.axvline(x=theory_r_max, color='green', linestyle=':', alpha=0.5, linewidth=1.5,
               label=f'Theoretical range end: {theory_r_max:.2f}')
    
    # y轴范围固定在[0, 1.1]以显示概率
    plt.ylim(-0.05, 1.1)
    
    # 添加水平参考线
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.axhline(y=1, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # 添加垂直参考线，标记理论曲线结束点
    plt.axvline(x=theory_r_max, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # 在垂直参考线处添加文本
    plt.text(theory_r_max, 1.05, f'{theory_r_max:.2f}', color='green', 
             fontsize=10, ha='center', va='bottom')
    
    # 添加图例说明理论范围
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='green', linestyle=':', linewidth=1.5)]
    plt.legend(custom_lines, [f'Theoretical range: [0, {theory_r_max:.2f}]'], 
               fontsize=10, loc='lower right')
    
    # 重新添加主图例（防止被覆盖）
    plt.legend(fontsize=12, loc='best')
    
    # 布局调整
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    print(f"图形已保存到 {savepath}")

def main():
    """
    主函数
    """
    # ========== 设置参数 ==========
    # 理论参数（根据图片中的公式）
    Ek = 1.0
    k = 2.0
    xi_f = 10.0  # 折叠状态长度
    xi_u = k*xi_f  # 展开状态长度
    E = Ek*xi_f
    
    # 实验数据归一化参数
    N = 1.0     # 归一化因子（domain全部打开时n=1）
    
    # 文件路径
    exp_r_file = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results/r_values.csv'
    exp_n_file = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results/n_values.csv'
    savepath = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results/compare_n_r_curves.png'
    
    params = {
        'E': E,
        'xi_f': xi_f,
        'xi_u': xi_u
    }
    
    # ========== 处理实验数据 ==========
    print("=" * 60)
    print("开始处理实验数据...")
    r_data, n_data = read_csv_files(exp_r_file, exp_n_file)
    
    if r_data is None or n_data is None:
        print("无法读取实验数据，程序退出。")
        return
    
    exp_r_uniform, exp_normalized_avg, exp_normalized_curves = process_experimental_data(r_data, n_data, N)
    
    if exp_r_uniform is None:
        print("实验数据处理失败，程序退出。")
        return
    
    print(f"实验数据处理完成：{len(exp_normalized_curves)}条轨迹")
    
    # ========== 计算理论曲线 ==========
    print("\n" + "=" * 60)
    print("开始计算理论曲线（使用fsolve）...")
    print(f"理论参数: E={E}, ξ_f={xi_f}, ξ_u={xi_u}")
    print(f"理论求解范围: [0, {0.95*xi_u:.2f}] (0.95*ξ_u)")
    
    # 使用fsolve方法计算理论曲线（固定范围：[0, 0.95*xi_u]）
    theory_r, theory_f, theory_pf, theory_puf = calculate_theoretical_curves_fsolve(
        E, xi_f, xi_u, num_points=1000
    )
    
    # 统计有效点
    valid_count = np.sum(~np.isnan(theory_puf))
    print(f"理论曲线计算完成，有效点: {valid_count}/{len(theory_r)}")
    
    if valid_count == 0:
        print("理论曲线计算失败，没有找到有效解。")
        return
    
    # ========== 输出理论结果摘要 ==========
    print("\n理论结果摘要:")
    print(f"  理论曲线r范围: [0, {np.max(theory_r):.2f}] (0.95*ξ_u)")
    print(f"  最大未折叠概率: {np.nanmax(theory_puf):.4f}")
    print(f"  最小未折叠概率: {np.nanmin(theory_puf):.4f}")
    print(f"  平均未折叠概率: {np.nanmean(theory_puf):.4f}")
    print(f"  NaN值比例: {100*(1 - valid_count/len(theory_puf)):.1f}%")
    
    # ========== 输出实验数据范围 ==========
    print("\n实验数据范围:")
    print(f"  实验数据r范围: [{np.min(exp_r_uniform):.3f}, {np.max(exp_r_uniform):.3f}]")
    print(f"  理论曲线r范围: [0, {np.max(theory_r):.3f}]")
    print(f"  共同比较范围: [0, {min(np.max(exp_r_uniform), np.max(theory_r)):.3f}]")
    
    # ========== 整合可视化 ==========
    print("\n" + "=" * 60)
    print("生成整合图表...")
    visualize_combined_curves(
        exp_r_uniform, exp_normalized_avg, exp_normalized_curves,
        theory_r, theory_puf, xi_u, params, savepath
    )
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)

if __name__ == "__main__":
    # 运行主函数
    main()
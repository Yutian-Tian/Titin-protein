import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.optimize import fsolve

# ========== 程序一的部分：读取CSV数据并处理 ==========
def read_csv_files(file1_path, file2_path):
    """
    读取两个CSV文件
    """
    try:
        # 读取CSV文件
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        
        return df1, df2
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None
    except Exception as e:
        print(f"Error reading files: {e}")
        return None, None

def process_experimental_data(df1, df2, N=1.0):
    """
    处理实验数据，将n除以N归一化，并计算平均值
    
    返回:
    r_uniform: 统一的r值网格
    normalized_avg: 归一化后的平均值曲线
    all_normalized_curves: 所有归一化后的原始曲线
    """
    if df1 is None or df2 is None:
        print("DataFrames are empty. Cannot process.")
        return None, None, None
    
    # 获取列数，确保两个文件列数相同
    num_columns = min(len(df1.columns), len(df2.columns))
    
    # 用于平均值计算的数据（所有原始数据）
    all_curves_data = []
    
    # 收集所有r值范围，用于生成统一的插值r数组
    all_r_min = []
    all_r_max = []
    
    # 首先处理所有数据，准备用于平均值计算
    for i in range(num_columns):
        r = df1.iloc[:, i].dropna().values
        n = df2.iloc[:, i].dropna().values
        
        # 确保r和n长度相同
        min_len = min(len(r), len(n))
        r = r[:min_len]
        n = n[:min_len]
        
        # 保存所有原始数据
        all_curves_data.append((r.copy(), n.copy()))
        
        # 收集r值范围
        if len(r) > 0:
            all_r_min.append(np.min(r))
            all_r_max.append(np.max(r))
    
    if len(all_curves_data) == 0:
        print("No valid experimental data found.")
        return None, None, None
    
    # ========== 平均值计算部分 ==========
    # 确定用于平均值计算的r值范围
    r_min = min(all_r_min)
    r_max = max(all_r_max)
    
    # 生成用于平均值计算的统一r值数组
    r_uniform = np.linspace(r_min, r_max, 10000)
    nsum = np.zeros_like(r_uniform)
    count = np.zeros_like(r_uniform)
    
    # 对所有数据进行插值并求和
    for r_orig, n_orig in all_curves_data:
        # 线性插值
        interp_func = interpolate.interp1d(r_orig, n_orig, kind='linear', 
                                          bounds_error=False, fill_value=np.nan)
        
        # 在统一r值下计算插值n值
        n_interp = interp_func(r_uniform)
        
        # 累加有效值
        valid_mask = ~np.isnan(n_interp)
        nsum[valid_mask] += n_interp[valid_mask]
        count[valid_mask] += 1
    
    # 计算平均值
    with np.errstate(divide='ignore', invalid='ignore'):
        n_avg = np.where(count > 0, nsum / count, np.nan)
    
    # 归一化：除以N
    normalized_avg = n_avg / N
    
    # 收集所有归一化后的原始曲线（用于绘制灰色半透明线）
    all_normalized_curves = []
    for r_orig, n_orig in all_curves_data:
        normalized_n = n_orig / N
        all_normalized_curves.append((r_orig.copy(), normalized_n.copy()))
    
    return r_uniform, normalized_avg, all_normalized_curves

# ========== 程序二的部分：理论计算 ==========
def iterative_solution_simple(E, xi, N, L, num_points=1000, max_iter=1000, tolerance=1e-6):
    """
    使用简单的迭代方法求解f
    
    参数:
    E, xi, N, L: 公式参数
    num_points: 计算点的数量
    max_iter: 最大迭代次数
    tolerance: 收敛容差
    """
    
    # 生成r值数组
    r_min = 0.0
    r_max = L
    r_values = np.linspace(r_min, r_max, num_points)
    
    # 存储结果的数组
    f_values = np.zeros_like(r_values)
    pf_values = np.zeros_like(r_values)
    puf_values = np.zeros_like(r_values)
    
    # 迭代求解每个r对应的f值
    for i, r in enumerate(r_values):
        # 初始猜测值
        f_guess = 0.1
        converged = False
        
        for iteration in range(max_iter):
            # 公式1: p_f(f) = 1/(1+exp(-E+f*xi))
            if f_guess *  xi <= 700:
                pf = 1.0 / (1.0 + np.exp(-E + f_guess * xi))
            else:
                pf = 0.0

            # 公式2: L_c(f) = L - p_f * N * xi
            Lc = L - pf * N * xi
            
            # 检查Lc是否有效
            if Lc <= 0 or r >= Lc:
                f_values[i] = np.nan
                pf_values[i] = np.nan
                puf_values[i] = np.nan
                converged = False
                break
            
            # 添加数值检查
            ratio = r/Lc
            if ratio >= 0.99:  # 接近极限时使用近似或设为NaN
                f_new = np.nan
            else:
                f_new = 0.25 * ((1 - ratio)**(-2) - 1 + 4*ratio)
            
            # 检查收敛
            if abs(f_new - f_guess) < tolerance:
                f_values[i] = f_new
                pf_values[i] = pf
                puf_values[i] = 1 - pf
                converged = True
                break
            
            f_guess = f_new
        
        if not converged:
            f_values[i] = np.nan
            pf_values[i] = np.nan
            puf_values[i] = np.nan
    
    return r_values, f_values, pf_values, puf_values

# ========== 整合可视化部分 ==========
def visualize_combined_curves(exp_r, exp_avg, exp_curves, theory_r, theory_puf, params, savepath):
    """
    在同一张图中绘制实验数据（归一化后）和理论曲线
    
    参数:
    exp_r: 实验数据统一的r值网格
    exp_avg: 归一化后的实验平均值
    exp_curves: 所有归一化后的实验原始曲线
    theory_r: 理论计算的r值
    theory_puf: 理论计算的p_uf值
    params: 参数字典
    savepath: 保存路径
    """
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # ========== 绘制实验数据 ==========
    # 1. 绘制所有原始实验曲线（灰色半透明）
    GRAY_COLOR = 'gray'
    GRAY_ALPHA = 0.1
    GRAY_LINEWIDTH = 1.0
    
    for r_orig, n_orig in exp_curves:
        if len(r_orig) > 2:  # 至少需要3个点才能绘制曲线
            # 线性插值
            interp_func = interpolate.interp1d(r_orig, n_orig, kind='linear', 
                                              bounds_error=False, fill_value=np.nan)
            
            # 生成该曲线的r值网格
            r_curve = np.linspace(np.min(r_orig), np.max(r_orig), 500)
            n_interp = interp_func(r_curve)
            
            # 绘制插值曲线（半透明灰线）
            valid_mask = ~np.isnan(n_interp)
            if np.any(valid_mask):
                plt.plot(r_curve[valid_mask], n_interp[valid_mask], 
                        color=GRAY_COLOR, linewidth=GRAY_LINEWIDTH, alpha=GRAY_ALPHA)
    
    # 2. 绘制实验平均值曲线（蓝色）
    valid_exp_mask = ~np.isnan(exp_avg)
    if np.any(valid_exp_mask):
        plt.plot(exp_r[valid_exp_mask], exp_avg[valid_exp_mask], 
                color='blue', linewidth=2.5, linestyle='-',
                label=f'Numerical Simulation $n_u/N$ (n={len(exp_curves)})', alpha=0.8)
    
    # ========== 绘制理论曲线 ==========
    valid_theory_mask = ~np.isnan(theory_puf)
    if np.any(valid_theory_mask):
        plt.plot(theory_r[valid_theory_mask], theory_puf[valid_theory_mask], 
                color='green', linewidth=3.0, linestyle='--',
                label='Theoretical $p_u (r)$', alpha=0.9)
    
    # ========== 设置图形属性 ==========
    plt.xlabel('$r$', fontsize=16)
    plt.ylabel('Probability of unfolded domains $p_u$', fontsize=16)
    
    # 设置标题，包含参数信息
    title = f'Experimental vs Theoretical Curves\n'
    plt.title(title, fontsize=18, pad=20)
    
    # 添加图例
    plt.legend(fontsize=12, loc='best')
    
    # 设置网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 设置刻度
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=14)
    
    # 设置坐标轴范围
    # 确定所有数据的有效范围
    all_r_min = []
    all_r_max = []
    all_y_min = []
    all_y_max = []
    
    # 实验数据范围
    if np.any(valid_exp_mask):
        all_r_min.append(np.min(exp_r[valid_exp_mask]))
        all_r_max.append(np.max(exp_r[valid_exp_mask]))
        all_y_min.append(np.min(exp_avg[valid_exp_mask]))
        all_y_max.append(np.max(exp_avg[valid_exp_mask]))
    
    # 理论数据范围
    if np.any(valid_theory_mask):
        all_r_min.append(np.min(theory_r[valid_theory_mask]))
        all_r_max.append(np.max(theory_r[valid_theory_mask]))
        all_y_min.append(np.min(theory_puf[valid_theory_mask]))
        all_y_max.append(np.max(theory_puf[valid_theory_mask]))
    
    if all_r_min and all_r_max and all_y_min and all_y_max:
        x_min, x_max = min(all_r_min), max(all_r_max)
        y_min, y_max = min(all_y_min), max(all_y_max)
        
        # 自动调整x轴范围，留5%边距
        x_margin = 0.05 * (x_max - x_min)
        plt.xlim(x_min - x_margin, x_max + x_margin)
        
        # 自动调整y轴范围，留10%边距
        y_margin = 0.1 * (y_max - y_min)
        plt.ylim(max(0, y_min - y_margin), min(1, y_max + y_margin))
   
    
    # 布局调整
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    print(f"整合图形已保存到 {savepath}")
    

def main():
    """
    主函数
    """
    # ========== 设置参数 ==========
    # 理论参数
    E = 3.0     # 能量参数
    xi = 30.0   # ξ参数
    N = 10.0    # N参数（用于归一化）
    L = 305.0   # L参数
    
    # 文件路径
    exp_r_file = '/home/tyt/project/Single-chain/opt+R/Rand_length/column_format/r_values.csv'
    exp_n_file = '/home/tyt/project/Single-chain/opt+R/Rand_length/column_format/n_values.csv'
    savepath = '/home/tyt/project/Single-chain/opt+R/Rand_length/column_format/combined_n_r_curves.png'
    
    params = {
        'E': E,
        'xi': xi,
        'N': N,
        'L': L
    }
    
    # ========== 处理实验数据 ==========
    print("开始处理实验数据...")
    df1, df2 = read_csv_files(exp_r_file, exp_n_file)
    
    if df1 is None or df2 is None:
        print("无法读取实验数据，程序退出。")
        return
    
    exp_r_uniform, exp_normalized_avg, exp_normalized_curves = process_experimental_data(df1, df2, N)
    
    if exp_r_uniform is None:
        print("实验数据处理失败，程序退出。")
        return
    
    print(f"实验数据处理完成：{len(exp_normalized_curves)}条曲线")
    
    # ========== 计算理论曲线 ==========
    print("开始计算理论曲线...")
    print(f"理论参数: E={E}, ξ={xi}, N={N}, L={L}")
    
    # 使用迭代方法计算
    theory_r, theory_f, theory_pf, theory_puf = iterative_solution_simple(E, xi, N, L, num_points=1000)
    
    # 统计有效点
    valid_count = np.sum(~np.isnan(theory_f))
    print(f"理论曲线计算完成，有效点: {valid_count}/{len(theory_r)}")
    
    if valid_count == 0:
        print("理论曲线计算失败，没有找到有效解。")
        return
    
    # ========== 整合可视化 ==========
    print("生成整合图表...")
    visualize_combined_curves(
        exp_r_uniform, exp_normalized_avg, exp_normalized_curves,
        theory_r, theory_puf, params, savepath
    )
    
    # ========== 可选：保存数据 ==========
    save_data = input("是否保存数据到CSV文件？(y/n): ").lower()
    if save_data == 'y':
        # 保存实验数据
        exp_data = pd.DataFrame({
            'r_exp': exp_r_uniform,
            'n/N_exp': exp_normalized_avg
        })
        exp_filename = f"experimental_data_E{params['E']}_xi{params['xi']}_N{params['N']}_L{params['L']}.csv"
        exp_data.to_csv(exp_filename, index=False)
        print(f"实验数据已保存到 {exp_filename}")
        
        # 保存理论数据
        theory_data = pd.DataFrame({
            'r_theory': theory_r,
            'f': theory_f,
            'p_f': theory_pf,
            'p_uf': theory_puf
        })
        theory_filename = f"theoretical_data_E{params['E']}_xi{params['xi']}_N{params['N']}_L{params['L']}.csv"
        theory_data.to_csv(theory_filename, index=False)
        print(f"理论数据已保存到 {theory_filename}")
    
    print('\n程序执行完成!')

if __name__ == "__main__":
    # 运行主函数
    main()
"""
比对数值模拟与2-state理论（使用蒙特卡洛方法计算ξ_f分布平均，修正高斯分布抽样）
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

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

# ========== WLC自由能函数（保持不变） ==========
def F_WLC_high_precision(r, L_c):
    """
    高精度计算WLC自由能，避免数值不稳定
    
    公式:
    F_WLC(x, L_c) = (π²/(2L_c)) * (1 - x²) + (2L_c/(π(1-x²)))
    其中 x = r/L_c
    
    参数:
    r: 端端距离
    L_c: 轮廓长度
    
    返回:
    F: 自由能
    """
    # 确保L_c > 0
    if L_c <= 0:
        return np.inf
    
    # 计算x = r/L_c
    x = r / L_c
    
    # 确保x在有效范围内 (|x| < 1)
    if abs(x) >= 1 - 1e-10:  # 添加小容差
        return np.inf
    
    # 使用高精度计算，避免小分母
    if abs(1 - x**2) < 1e-10:
        # 如果分母接近0，返回大值
        return 1e10
    
    # 计算F_WLC
    term1 = (np.pi**2) / (2 * L_c) * (1 - x**2)
    term2 = (2 * L_c) / (np.pi * (1 - x**2))
    
    return term1 + term2

def single_ensemble_average_high_precision(r, xi_f, E, xi_u, beta=1.0):
    """
    高精度计算单个ξ_f下的系综平均 ⟨n⟩(r; ξ_f)
    
    公式:
    ⟨n⟩(r; ξ_f) = exp[-βΔ(r; ξ_f)] / (1 + exp[-βΔ(r; ξ_f)])
    Δ(r; ξ_f) = F_WLC(r, ξ_u) - F_WLC(r, xi_f) + ΔE
    
    参数:
    r: 端端距离
    xi_f: 折叠状态轮廓长度
    E: ΔE 能量差
    xi_u: 展开状态轮廓长度
    beta: β 参数
    
    返回:
    p_u: 打开概率
    """
    # 计算F_WLC(r, xi_u) 和 F_WLC(r, xi_f)
    F_unfolded = F_WLC_high_precision(r, xi_u)
    F_folded = F_WLC_high_precision(r, xi_f)
    
    # 如果自由能为无穷大，处理边界情况
    if np.isinf(F_unfolded) and np.isinf(F_folded):
        # 两个状态都无效，返回中间值
        return 0.5
    elif np.isinf(F_unfolded):
        # 展开状态无效，返回0
        return 0.0
    elif np.isinf(F_folded):
        # 折叠状态无效，返回1
        return 1.0
    
    # 计算Δ(r; ξ_f)
    Delta = F_unfolded - F_folded + E
    
    # 计算打开概率，使用数值稳定的sigmoid函数
    # 避免指数溢出
    if -beta * Delta > 700:
        p_u = 1.0
    elif -beta * Delta < -700:
        p_u = 0.0
    else:
        exp_term = np.exp(-beta * Delta)
        p_u = exp_term / (1 + exp_term)
    
    return p_u

# ========== 修正的蒙特卡洛方法：正确模拟高斯分布 ==========
def monte_carlo_double_average_corrected(r, mu_xi_f, sigma_xi_f, Ek, uk, beta=1.0, n_samples=10000):
    """
    使用蒙特卡洛方法计算双重平均，正确模拟ξ_f的高斯分布
    
    注意：ξ_f必须为正。对于高斯分布，我们需要处理负值情况。
    物理上，ξ_f > 0，所以我们有两种选择：
    1. 使用截断高斯分布（truncated normal distribution）
    2. 使用重采样方法：如果抽样到负值，则重新抽样
    
    这里我们采用方法2，确保所有样本都为正。
    
    蒙特卡洛公式: ¯n(r) ≈ (1/N) Σ_{i=1}^{N} ⟨n⟩(r; ξ_f^{(i)})
    其中 ξ_f^{(i)} 是从高斯分布中抽取的正样本
    
    参数:
    r: 端端距离
    mu_xi_f: ξ_f分布的均值
    sigma_xi_f: ξ_f分布的标准差
    Ek: 能量参数 (E = Ek * xi_f^3)
    uk: 展开状态与折叠状态的比率 (xi_u = uk * xi_f)
    beta: β 参数
    n_samples: 蒙特卡洛样本数
    
    返回:
    p_u_avg: 平均打开概率
    std_err: 标准误差
    """
    # 从高斯分布中抽取样本，确保所有样本为正
    xi_f_samples = []
    attempts = 0
    max_attempts = n_samples * 100  # 最大尝试次数，防止无限循环
    
    while len(xi_f_samples) < n_samples and attempts < max_attempts:
        # 从高斯分布中抽取一个样本
        sample = np.random.normal(mu_xi_f, sigma_xi_f)
        
        # 只接受正样本
        if sample > 0:
            xi_f_samples.append(sample)
        
        attempts += 1
    
    # 检查是否收集到足够样本
    if len(xi_f_samples) < n_samples:
        print(f"警告: 只收集到 {len(xi_f_samples)} 个正样本（目标: {n_samples}）")
        if len(xi_f_samples) == 0:
            return 0.0, 0.0
        # 使用收集到的样本，即使数量不足
        n_samples = len(xi_f_samples)
    
    xi_f_samples = np.array(xi_f_samples)
    
    # 计算每个样本的⟨n⟩(r; ξ_f)
    p_u_values = np.zeros(n_samples)
    
    for i, xi_f in enumerate(xi_f_samples):
        # 计算当前ξ_f对应的E值
        E_current = Ek * (xi_f**3)
        
        # 计算xi_u
        xi_u = uk * xi_f
        
        # 计算单个系综平均
        p_u = single_ensemble_average_high_precision(r, xi_f, E_current, xi_u, beta)
        p_u_values[i] = p_u
    
    # 计算蒙特卡洛平均值和标准误差
    p_u_avg = np.mean(p_u_values)
    p_u_std = np.std(p_u_values)
    std_err = p_u_std / np.sqrt(n_samples)
    
    return p_u_avg, std_err

# ========== 方法2：使用截断高斯分布 ==========
def monte_carlo_truncated_normal(r, mu_xi_f, sigma_xi_f, Ek, uk, beta=1.0, n_samples=10000):
    """
    使用截断高斯分布的蒙特卡洛方法
    这种方法更正确，因为它考虑了ξ_f必须为正的物理约束
    """
    # 计算截断高斯分布的参数
    # 对于截断在[0, ∞)的高斯分布，我们需要调整均值和方差
    # 但为了简单，我们使用接受-拒绝法从原始高斯分布中抽样
    
    # 使用更高效的方法：直接从正态分布抽样，然后丢弃负值
    # 这是接受-拒绝法的一种简单形式
    
    # 计算从原始高斯分布中抽到正值的概率
    # 这等于原始高斯分布在(0, ∞)的累积概率
    prob_positive = 1.0 - norm.cdf(0, loc=mu_xi_f, scale=sigma_xi_f)
    
    # 如果需要n_samples个正样本，我们需要从原始分布中抽取大约n_samples/prob_positive个样本
    expected_samples_needed = int(n_samples / max(prob_positive, 0.01))
    
    # 从原始高斯分布中抽取样本
    raw_samples = np.random.normal(mu_xi_f, sigma_xi_f, expected_samples_needed * 2)
    
    # 选择正样本
    positive_samples = raw_samples[raw_samples > 0]
    
    # 如果正样本不足，抽取更多
    while len(positive_samples) < n_samples:
        additional_samples = np.random.normal(mu_xi_f, sigma_xi_f, n_samples)
        positive_samples = np.concatenate([positive_samples, additional_samples[additional_samples > 0]])
    
    # 取前n_samples个正样本
    xi_f_samples = positive_samples[:n_samples]
    
    # 计算每个样本的⟨n⟩(r; ξ_f)
    p_u_values = np.zeros(n_samples)
    
    for i, xi_f in enumerate(xi_f_samples):
        # 计算当前ξ_f对应的E值
        E_current = Ek * (xi_f**3)
        
        # 计算xi_u
        xi_u = uk * xi_f
        
        # 计算单个系综平均
        p_u = single_ensemble_average_high_precision(r, xi_f, E_current, xi_u, beta)
        p_u_values[i] = p_u
    
    # 计算蒙特卡洛平均值和标准误差
    p_u_avg = np.mean(p_u_values)
    p_u_std = np.std(p_u_values)
    std_err = p_u_std / np.sqrt(n_samples)
    
    return p_u_avg, std_err

def calculate_theoretical_curves_monte_carlo_corrected(mu_xi_f, sigma_xi_f, Ek, uk, beta=1.0, num_points=200, n_mc_samples=5000):
    """
    使用修正的蒙特卡洛方法计算考虑ξ_f分布的理论曲线
    
    参数:
    mu_xi_f: ξ_f分布的均值
    sigma_xi_f: ξ_f分布的标准差
    Ek: 能量参数 (E = Ek * xi_f^3)
    uk: 展开状态与折叠状态的比率 (xi_u = uk * xi_f)
    beta: β 参数
    num_points: r值点数
    n_mc_samples: 每个r点的蒙特卡洛样本数
    
    返回:
    r_values: r值数组
    p_u_values: 平均打开概率数组
    p_u_errors: 蒙特卡洛标准误差数组
    """
    # 确定r值范围
    # 使用均值xi_f计算最大r值，但不超过0.99*uk*mu_xi_f以避免数值问题
    r_max = min(0.99 * uk * mu_xi_f, 50)  # 限制最大r值
    r_min = 0.01  # 从0.01开始避免除以0
    
    # 创建r值数组（使用对数间隔在r较小时更密集采样）
    r_values = np.logspace(np.log10(r_min), np.log10(r_max), num_points)
    
    # 存储结果的数组
    p_u_values = np.zeros_like(r_values)
    p_u_errors = np.zeros_like(r_values)
    
    print("开始使用修正的蒙特卡洛方法计算考虑ξ_f分布的理论曲线...")
    print(f"r值范围: [{r_min:.3f}, {r_max:.3f}]")
    print(f"采样点数: {num_points}")
    print(f"每个点的蒙特卡洛样本数: {n_mc_samples}")
    print(f"ξ_f分布: N(μ={mu_xi_f}, σ={sigma_xi_f})")
    
    # 计算从高斯分布中抽到正值的概率
    prob_positive = 1.0 - norm.cdf(0, loc=mu_xi_f, scale=sigma_xi_f)
    print(f"ξ_f为正的概率: {prob_positive:.6f}")
    
    for i, r in enumerate(r_values):
        # 使用截断高斯分布的蒙特卡洛方法
        p_u_avg, std_err = monte_carlo_truncated_normal(r, mu_xi_f, sigma_xi_f, Ek, uk, beta, n_mc_samples)
        p_u_values[i] = p_u_avg
        p_u_errors[i] = std_err
        
        # 显示进度
        if (i+1) % max(1, num_points//20) == 0:
            print(f"  进度: {i+1}/{num_points} ({(i+1)/num_points*100:.1f}%), "
                  f"r={r:.3f}, p_u={p_u_avg:.4f} ± {std_err:.6f}")
    
    print(f"修正的蒙特卡洛计算完成")
    
    return r_values, p_u_values, p_u_errors

# ========== 可视化函数（保持不变） ==========
def visualize_combined_curves_with_error(exp_r, exp_avg, exp_curves, theory_r, theory_puf, theory_errors, params, savepath):
    """
    在同一张图中绘制实验数据（归一化后）和理论曲线（带误差带）
    """
    # 创建图形
    plt.figure(figsize=(14, 10))
    
    # ========== 绘制实验数据 ==========
    # 1. 绘制所有原始实验曲线（灰色半透明）
    GRAY_COLOR = 'gray'
    GRAY_ALPHA = 0.1
    GRAY_LINEWIDTH = 1.0
    
    for r_orig, n_orig in exp_curves:
        if len(r_orig) > 2:
            interp_func = interpolate.interp1d(r_orig, n_orig, kind='linear', 
                                              bounds_error=False, fill_value=np.nan)
            
            r_curve = np.linspace(np.min(r_orig), np.max(r_orig), 500)
            n_interp = interp_func(r_curve)
            
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
    
    # ========== 绘制理论曲线（带误差带） ==========
    valid_theory_mask = ~np.isnan(theory_puf)
    if np.any(valid_theory_mask):
        plt.plot(theory_r[valid_theory_mask], theory_puf[valid_theory_mask], 
                color='green', linewidth=3.0, linestyle='-',
                label='Theoretical $p_u (r)$ (Monte Carlo)', alpha=0.9)
        
        error_band = 2.0 * theory_errors[valid_theory_mask]
        plt.fill_between(theory_r[valid_theory_mask], 
                         theory_puf[valid_theory_mask] - error_band,
                         theory_puf[valid_theory_mask] + error_band,
                         color='green', alpha=0.2, label='Monte Carlo 95% CI')
    
    # ========== 设置图形属性 ==========
    plt.xlabel('End-to-end distance $r$', fontsize=16)
    plt.ylabel('Unfolded probability $p_u$', fontsize=16)
    
    title = f'Numerical Simulation vs Two-State Model with ξ_f Distribution\n'
    title += f'Monte Carlo (Truncated Gaussian): μ_ξ_f={params["mu_xi_f"]}, σ_ξ_f={params["sigma_xi_f"]}, Ek={params["Ek"]}, uk={params["uk"]}'
    plt.title(title, fontsize=14, pad=20)
    
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=14)
    
    # 设置坐标轴范围
    all_r_min = []
    all_r_max = []
    all_y_min = []
    all_y_max = []
    
    if np.any(valid_exp_mask):
        all_r_min.append(np.min(exp_r[valid_exp_mask]))
        all_r_max.append(np.max(exp_r[valid_exp_mask]))
        all_y_min.append(np.min(exp_avg[valid_exp_mask]))
        all_y_max.append(np.max(exp_avg[valid_exp_mask]))
    
    if np.any(valid_theory_mask):
        all_r_min.append(np.min(theory_r[valid_theory_mask]))
        all_r_max.append(np.max(theory_r[valid_theory_mask]))
        error_band = 2.0 * theory_errors[valid_theory_mask]
        all_y_min.append(np.min(theory_puf[valid_theory_mask] - error_band))
        all_y_max.append(np.max(theory_puf[valid_theory_mask] + error_band))
    
    if all_r_min and all_r_max and all_y_min and all_y_max:
        x_min, x_max = min(all_r_min), max(all_r_max)
        y_min, y_max = min(all_y_min), max(all_y_max)
        
        x_margin = 0.05 * (x_max - x_min)
        plt.xlim(x_min - x_margin, x_max + x_margin)
        
        y_margin = 0.1 * (y_max - y_min)
        plt.ylim(max(0, y_min - y_margin), min(1.1, y_max + y_margin))
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    print(f"整合图形已保存到 {savepath}")

def main():
    """
    主函数
    """
    # ========== 设置参数 ==========
    Ek = 0.02          # 能量参数
    uk = 5.0           # 展开状态与折叠状态的比率
    mu_xi_f = 5.0      # ξ_f分布的均值
    sigma_xi_f = 0.5   # ξ_f分布的标准差
    beta = 1.0         # β参数
    
    # 蒙特卡洛参数
    n_mc_samples = 10000  # 每个r点的蒙特卡洛样本数
    
    # 实验数据归一化参数
    N = 1.0            # 归一化因子
    
    # 文件路径
    exp_r_file = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results/r_values.csv'
    exp_n_file = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results/n_values.csv'
    savepath = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results/compare_n_r_curves_monte_carlo_corrected.png'
    
    params = {
        'mu_xi_f': mu_xi_f,
        'sigma_xi_f': sigma_xi_f,
        'Ek': Ek,
        'uk': uk,
        'beta': beta,
        'n_mc_samples': n_mc_samples
    }
    
    # ========== 处理实验数据 ==========
    print("=" * 60)
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
    
    # ========== 使用修正的蒙特卡洛方法计算理论曲线 ==========
    print("\n" + "=" * 60)
    print("开始使用修正的蒙特卡洛方法计算理论曲线...")
    print(f"理论参数: μ_ξ_f={mu_xi_f}, σ_ξ_f={sigma_xi_f}, Ek={Ek}, uk={uk}, β={beta}")
    print(f"蒙特卡洛参数: {n_mc_samples} 个样本/点")
    
    # 使用修正的蒙特卡洛方法计算理论曲线
    theory_r, theory_puf, theory_errors = calculate_theoretical_curves_monte_carlo_corrected(
        mu_xi_f, sigma_xi_f, Ek, uk, beta, num_points=200, n_mc_samples=n_mc_samples
    )
    
    # 统计有效点
    valid_count = np.sum(~np.isnan(theory_puf))
    print(f"理论曲线计算完成，有效点: {valid_count}/{len(theory_r)}")
    
    if valid_count == 0:
        print("理论曲线计算失败，没有找到有效解。")
        return
    
    # ========== 输出理论结果摘要 ==========
    print("\n理论结果摘要:")
    print(f"  r范围: [{np.min(theory_r):.3f}, {np.max(theory_r):.3f}]")
    print(f"  最大未折叠概率: {np.max(theory_puf):.6f}")
    print(f"  最小未折叠概率: {np.min(theory_puf):.6f}")
    print(f"  平均未折叠概率: {np.mean(theory_puf):.6f}")
    print(f"  平均标准误差: {np.mean(theory_errors):.6f}")
    print(f"  最大标准误差: {np.max(theory_errors):.6f}")
    
    # ========== 整合可视化 ==========
    print("\n" + "=" * 60)
    print("生成整合图表...")
    visualize_combined_curves_with_error(
        exp_r_uniform, exp_normalized_avg, exp_normalized_curves,
        theory_r, theory_puf, theory_errors, params, savepath
    )
    
    # ========== 保存理论计算结果 ==========
    theory_df = pd.DataFrame({
        'r': theory_r,
        'p_u': theory_puf,
        'p_u_error': theory_errors,
        'p_u_lower_95': theory_puf - 2*theory_errors,
        'p_u_upper_95': theory_puf + 2*theory_errors
    })
    theory_save_path = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results/theoretical_results_monte_carlo_corrected.csv'
    theory_df.to_csv(theory_save_path, index=False)
    print(f"\n理论计算结果已保存到: {theory_save_path}")
    
    print("\n" + "=" * 60)
    print("程序执行完成！")
    print("=" * 60)

if __name__ == "__main__":
    # 设置随机种子以确保结果可重复
    np.random.seed(42)
    
    # 运行主函数
    main()
'''
单个domain的模拟和理论对比 Delta理论
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.optimize import fsolve

# ============ 字体路径 ============
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

# ============ 样式变量定义 ============
# 字体设置
font_family = 'Times New Roman'
font_weight = 'normal'
math_fontset = 'stix'
math_rm = 'Times New Roman'
math_it = 'Times New Roman:italic'
math_bf = 'Times New Roman:bold'

# 字体大小
title_fontsize = 35
label_fontsize = 35
tick_fontsize = 35
legend_fontsize = 25
legend_title_fontsize = 35

# 线宽和尺寸
axes_linewidth = 2
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 10
ytick_major_size = 10
grid_linewidth = 1
grid_alpha = 0.4
lines_linewidth = 5
lines_markersize = 15

# 刻度方向
xtick_direction = 'in'
ytick_direction = 'in'
xtick_top = True
ytick_right = True

# 图形设置
figure_dpi = 100
savefig_dpi = 300

# ============ 应用全局设置 ============
if os.path.exists(font_path):
    # 将字体文件添加到matplotlib的字体管理器中
    fm.fontManager.addfont(font_path)
    # 获取字体的属性
    font_prop = fm.FontProperties(fname=font_path)
    # 将字体的名称设置为默认字体
    plt.rcParams['font.family'] = font_prop.get_name()

plt.rcParams.update({
    'font.family': font_family,
    'mathtext.fontset': math_fontset,
    'mathtext.rm': math_rm,
    'mathtext.it': math_it,
    'mathtext.bf': math_bf,
    'font.weight': font_weight,
    'axes.titlesize': title_fontsize,
    'axes.labelsize': label_fontsize,
    'xtick.labelsize': tick_fontsize,
    'ytick.labelsize': tick_fontsize,
    'legend.fontsize': legend_fontsize,
    'legend.title_fontsize': legend_title_fontsize,
    'axes.linewidth': axes_linewidth,
    'xtick.major.width': xtick_major_width,
    'ytick.major.width': ytick_major_width,
    'xtick.major.size': xtick_major_size,
    'ytick.major.size': ytick_major_size,
    'grid.linewidth': grid_linewidth,
    'grid.alpha': grid_alpha,
    'lines.linewidth': lines_linewidth,
    'lines.markersize': lines_markersize,
    'figure.dpi': figure_dpi,
    'savefig.dpi': savefig_dpi,
    'xtick.direction': xtick_direction,
    'ytick.direction': ytick_direction,
    'xtick.top': xtick_top,
    'ytick.right': ytick_right,
})

# ============ 参数设置 ============
xi_f = 10.0  # 折叠态轮廓长度
xi_u = 20.0  # 展开态轮廓长度
DeltaE = 10.0  # 能量差 ΔE
DeltaEt = 24.5  # 理论部分使用的能量差
U0 = 20.0     # 周期性势阱深度 U0
rs = 7.0  # 反应距离
pi = np.pi
delta = 1/(2*np.pi)*np.arcsin(- DeltaE/(4*np.pi*U0))
f_min = 0.0
f_max = 31.0
N = 4.0 # 折叠域的数量
f_grid = 200
r_grid = 800
n_grid = 200

# ============ 辅助函数定义 ============
def pf_from_force(f):
    """根据力f计算折叠概率p_f"""
    # p_f(f) = 1 / (1 + exp(-ΔE + f·r_s))
    exponent = -DeltaEt + f * rs
    if exponent > 50:  # 避免溢出
        return 0.0
    elif exponent < -50:  # 避免下溢
        return 1.0
    else:
        return 1.0 / (1.0 + np.exp(exponent))

def force_from_r_and_Lc(r, Lc):
    """根据r和Lc计算力f"""
    # x = r / L_c(f)
    x = r / Lc
    
    # 避免数值问题
    if x >= 0.999:
        x = 0.999
    elif x <= 0:
        x = 0.001
    
    # f = -π²x/L_c² + 4x/[π(1-x²)²]
    term1 = -(pi**2 * x) / (Lc**2)
    term2 = (4 * x) / (pi * (1 - x**2)**2)
    return term1 + term2

def self_consistent_equations(f, r):
    """自洽方程组：定义需要满足的方程
    
    根据图片中的方程：
    1. p_f = 1/(1+exp(-ΔE + f·r_s))
    2. L_c = ξ_u - p_f(ξ_u - ξ_f)
    3. x = r/L_c
    4. f_calculated = -π²x/L_c² + 4x/[π(1-x²)²]
    
    我们需要 f = f_calculated
    所以方程是：f - f_calculated = 0
    """
    # 计算p_f
    pf = pf_from_force(f)
    
    # 计算L_c
    Lc = N*(xi_u - pf * (xi_u - xi_f))
    
    # 检查Lc是否有效
    if Lc <= 0:
        # 返回一个大的残差，表示不可行
        return 1e6
    
    # 计算x
    x = r / Lc
    
    # 避免数值问题
    if x >= 0.999:
        x = 0.999
    elif x <= 0:
        x = 0.001
    
    # 计算基于当前Lc和x的力
    f_calc = -(pi**2 * x) / (Lc**2) + (4 * x) / (pi * (1 - x**2)**2)
    
    # 返回残差：f - f_calc
    return f - f_calc

def solve_self_consistent(r, initial_guess=None):
    """使用fsolve求解自洽方程
    
    参数：
    r: 端到端距离
    initial_guess: 初始猜测值，如果为None则使用自适应猜测
    """
    # 如果没有提供初始猜测，则根据r值提供合理的初始猜测
    if initial_guess is None:
        if r < 5:
            f_guess = 0.1
        elif r < 10:
            f_guess = 1.0
        elif r < 15:
            f_guess = 5.0
        else:
            f_guess = 10.0
    else:
        f_guess = initial_guess
    
    try:
        # 使用fsolve求解
        f_solution = fsolve(self_consistent_equations, f_guess, args=(r,), full_output=True)
        
        # f_solution是一个元组：(解, 信息字典, 错误标志, 消息)
        f_result = f_solution[0][0]
        info = f_solution[1]  # 求解信息
        
        # 检查解是否有效
        if f_result < 0:
            # 如果力为负，尝试从0开始
            f_solution = fsolve(self_consistent_equations, 0.0, args=(r,), full_output=True)
            f_result = f_solution[0][0]
        
        # 获取最终解
        f_final = f_result
        
        # 计算对应的p_f和p_u
        pf = pf_from_force(f_final)
        p_u = 1.0 - pf
        
        return p_u, f_final, True
    except Exception as e:
        print(f"在r={r:.2f}处求解失败: {e}")
        # 返回边界值
        if r < xi_f:
            return 0.0, 0.0, False
        else:
            return 1.0, 0.0, False

# ============ 模拟部分（基于修正自由能最小化） ============

def Lc(n):
    """轮廓长度作为n的函数"""
    return N*xi_f + n * (xi_u - xi_f)

def x(r, n):
    """端到端距离与轮廓长度之比"""
    return r / Lc(n)

def F_WLC(r, n):
    """WLC自由能"""
    Lc_val = Lc(n)
    x_val = x(r, n)
    if x_val >= 1.0 or x_val < 0:  # 避免数值问题
        return np.inf
    term1 = (pi**2) / (2 * Lc_val) * (1 - x_val**2)
    term2 = (2 * Lc_val) / (pi * (1 - x_val**2))
    return term1 + term2

def U(n):
    """周期性势能项"""
    n1 = n + delta
    return DeltaE * n - U0 * np.cos(2 * pi * n1)

def F_total(r, n):
    """总自由能"""
    return F_WLC(r, n) + U(n)

def F_modified(r, n, f):
    """修正自由能：F(r,n) - f·r"""
    return F_total(r, n) - f * r

def simulate_minimize_F_modified(f_values, r_points=200, n_points=100):
    """对于每个给定的力f，通过扫描r和n最小化修正自由能F(r,n;f) = F_total(r,n) - f·r"""
    n_sim = []
    r_sim = []
    
    # 创建r和n的网格
    r_min = 0.0
    r_max = 0.95 * N * xi_u  # r的最大值
    r_grid = np.linspace(r_min, r_max, r_points)
    n_grid = np.linspace(0, N, n_points)
    
    for f in f_values:
        min_F = np.inf
        r_opt = 0.0
        n_opt = 0.0
        
        # 扫描r和n
        for r in r_grid:
            for n in n_grid:
                # 计算修正自由能
                F_mod = F_modified(r, n, f)
                
                # 检查是否有效且更小
                if np.isfinite(F_mod) and F_mod < min_F:
                    min_F = F_mod
                    r_opt = r
                    n_opt = n
        
        r_sim.append(r_opt)
        n_sim.append(n_opt)
    
    return np.array(r_sim), np.array(n_sim)

# ============ 理论部分（基于图片中的自洽方程，使用fsolve求解） ============
def theoretical_n_f_fsolve(r_values):
    """基于图片中自洽方程的理论解，使用fsolve求解
    
    修改：将上一次的结果作为新的初始猜测解进行迭代
    """
    n_theory = []
    f_theory = []
    
    # 用于跟踪收敛情况的变量
    convergence_info = []
    
    # 第一个点的初始猜测
    prev_f_guess = None
    
    for i, r in enumerate(r_values):
        # 使用上一次的结果作为初始猜测（如果可用）
        if prev_f_guess is not None:
            initial_guess = prev_f_guess
        else:
            initial_guess = None
        
        # 使用fsolve求解自洽方程
        p_u, f_val, success = solve_self_consistent(r, initial_guess)
        
        # 保存结果
        n_theory.append(p_u)
        f_theory.append(f_val)
        
        # 更新下一次的初始猜测
        prev_f_guess = f_val
        
        # 记录收敛信息
        convergence_info.append({
            'r': r,
            'p_u': p_u,
            'f': f_val,
            'success': success,
            'used_prev_guess': initial_guess is not None
        })
        
        # 显示进度
        if (i+1) % 20 == 0 or i == 0 or i == len(r_values)-1:
            guess_info = "使用前值" if initial_guess is not None else "自适应猜测"
            status = "成功" if success else "失败"
            print(f"  进度: {i+1}/{len(r_values)}, r={r:.2f}, p_u={p_u:.4f}, f={f_val:.4f}, {guess_info}, {status}")
    
    # 输出收敛统计
    successes = sum(1 for info in convergence_info if info['success'])
    prev_guesses = sum(1 for info in convergence_info if info['used_prev_guess'])
    print(f"\n收敛统计:")
    print(f"  总计算点数: {len(convergence_info)}")
    print(f"  成功点数: {successes}")
    print(f"  使用前值作为初始猜测的点数: {prev_guesses}")
    print(f"  成功率: {successes/len(convergence_info)*100:.1f}%")
    
    return np.array(n_theory), np.array(f_theory)

# ============ 主程序 ============
def main():
    # 创建r值数组
    r_min = 0.0
    r_max = 0.95 * N * xi_u  # 避免r=0和r>=Lc的边界问题
    r_values = np.linspace(r_min, r_max, r_grid)
    
    # 计算理论结果（基于图片中的自洽方程，使用fsolve）
    print("开始理论计算（使用fsolve，迭代初始猜测）...")
    n_theory, f_theory = theoretical_n_f_fsolve(r_values)
    print("理论计算完成")
    
    # 创建f值数组（模拟部分的输入）
    f_values = np.linspace(f_min, f_max, f_grid)  # 200个f值
    
    # 计算模拟结果（基于修正自由能最小化）
    print("开始模拟计算（修正自由能最小化）...")
    r_sim, n_sim = simulate_minimize_F_modified(f_values, r_points=r_grid, n_points=n_grid)
    print("模拟计算完成")
    
    # 为了将模拟结果（以f为自变量）转换为以r为自变量，我们需要建立f和r的对应关系
    # 由于模拟得到的是离散的(f, r, n)点，我们需要将这些点排序并插值到理论计算的r网格上
    
    # 首先，按r_sim排序
    sorted_indices = np.argsort(r_sim)
    r_sim_sorted = r_sim[sorted_indices]
    n_sim_sorted = n_sim[sorted_indices]
    f_sim_sorted = f_values[sorted_indices]  # 对应的f值
    
    # 插值到理论计算的r网格上
    from scipy import interpolate
    
    # 创建插值函数
    f_n_sim_interp = interpolate.interp1d(r_sim_sorted, n_sim_sorted, 
                                          bounds_error=False, fill_value=(n_sim_sorted[0], n_sim_sorted[-1]))
    f_f_sim_interp = interpolate.interp1d(r_sim_sorted, f_sim_sorted,
                                          bounds_error=False, fill_value=(f_sim_sorted[0], f_sim_sorted[-1]))
    
    # 在理论r值网格上计算插值结果
    n_sim_interp = f_n_sim_interp(r_values)/N
    f_sim_interp = f_f_sim_interp(r_values)
    
    # ============ 科研风格可视化 ============
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. n-r 曲线
    ax1 = axes[0]
    # 模拟曲线 - 使用变量中定义的线宽
    ax1.plot(r_values, n_sim_interp, color='red', linewidth=lines_linewidth, label='Simulation')
    # 理论曲线 - 使用变量中定义的线宽
    ax1.plot(r_values, n_theory, color='blue', linewidth=lines_linewidth, linestyle='--', label='Theory')
    
    # 使用变量中定义的字体大小和样式
    ax1.set_xlabel('End-to-end distance $r$', fontsize=label_fontsize)
    ax1.set_ylabel('Unfolding probability $p_u$', fontsize=label_fontsize)
    ax1.set_title('$p_u$ vs. distance $r$', fontsize=title_fontsize, pad=20)
    
    # 设置网格 - 使用变量中定义的线宽和透明度
    ax1.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    
    # 设置图例 - 使用变量中定义的字体大小
    ax1.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    
    # 设置坐标轴范围
    ax1.set_xlim(0, 0.95*N*xi_u)
    ax1.set_ylim(-0.1, 1.1)
    
    # 设置刻度（所有边框都有刻度，朝内） - 使用变量中定义的参数
    ax1.tick_params(axis='both', which='major', 
                   direction=xtick_direction,     # 刻度朝内
                   top=xtick_top,                 # 顶部显示刻度
                   right=ytick_right,             # 右侧显示刻度
                   bottom=True,                   # 底部显示刻度
                   left=True,                     # 左侧显示刻度
                   width=xtick_major_width,       # 刻度线宽
                   length=xtick_major_size)       # 刻度线长度
    
    ax1.tick_params(axis='both', which='minor',
                   direction=xtick_direction,     # 刻度朝内
                   top=xtick_top,                 # 顶部显示刻度
                   right=ytick_right,             # 右侧显示刻度
                   bottom=True,                   # 底部显示刻度
                   left=True,                     # 左侧显示刻度
                   width=xtick_major_width*0.75,  # 次刻度线宽稍小
                   length=xtick_major_size*0.5)   # 次刻度线长稍小
    
    # 开启次刻度
    ax1.minorticks_on()
    
    # 2. f-r 曲线
    ax2 = axes[1]
    # 模拟曲线 - 使用变量中定义的线宽
    ax2.plot(r_values, f_sim_interp, color='red', linewidth=lines_linewidth, label='Simulation')
    # 理论曲线 - 自洽方程解（fsolve）
    ax2.plot(r_values, f_theory, color='blue', linewidth=lines_linewidth, linestyle='--', label='Theory')
    
    # 使用变量中定义的字体大小和样式
    ax2.set_xlabel('End-to-end distance $r$', fontsize=label_fontsize)
    ax2.set_ylabel('Force $f$', fontsize=label_fontsize)
    ax2.set_title('Force $f$ vs. distance $r$', fontsize=title_fontsize, pad=20)
    
    # 设置网格 - 使用变量中定义的线宽和透明度
    ax2.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    
    # 设置图例 - 使用变量中定义的字体大小
    ax2.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    
    # 设置坐标轴范围
    ax2.set_xlim(0, N*xi_u)
    ax2.set_ylim(-1.5, 30)
    
    # 设置刻度（所有边框都有刻度，朝内） - 使用变量中定义的参数
    ax2.tick_params(axis='both', which='major', 
                   direction=xtick_direction,     # 刻度朝内
                   top=xtick_top,                 # 顶部显示刻度
                   right=ytick_right,             # 右侧显示刻度
                   bottom=True,                   # 底部显示刻度
                   left=True,                     # 左侧显示刻度
                   width=xtick_major_width,       # 刻度线宽
                   length=xtick_major_size)       # 刻度线长度
    
    ax2.tick_params(axis='both', which='minor',
                   direction=xtick_direction,     # 刻度朝内
                   top=xtick_top,                 # 顶部显示刻度
                   right=ytick_right,             # 右侧显示刻度
                   bottom=True,                   # 底部显示刻度
                   left=True,                     # 左侧显示刻度
                   width=xtick_major_width*0.75,  # 次刻度线宽稍小
                   length=xtick_major_size*0.5)   # 次刻度线长稍小
    
    # 开启次刻度
    ax2.minorticks_on()
    
    # 为两个子图添加边框强化 - 使用变量中定义的线宽
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(axes_linewidth)
    
    # 调整子图间距
    plt.subplots_adjust(wspace=0.25, top=0.9, bottom=0.1)
    
    plt.tight_layout()
    
    # 保存图形（多种格式）
    base_save_path = '/home/tyt/project/Single-chain/opt+force/2-state_comparation/results/theory_vs_numerical'
    
    # 保存为PNG（用于文档）- 使用变量中定义的分辨率
    plt.savefig(f'{base_save_path}.png', dpi=savefig_dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"\n图形已保存至: {base_save_path}.png")
    
    # ============ 输出数值结果示例 ============
    print("\n" + "="*80)
    print("参数设置:")
    print(f"  ξ_f = {xi_f}")
    print(f"  ξ_u = {xi_u}")
    print(f"  ΔE (模拟) = {DeltaE}")
    print(f"  ΔE (理论) = {DeltaEt}")
    print(f"  U0 = {U0}")
    print(f"  r_s = {rs}")
    print(f"\nr范围: {r_min:.2f} 到 {r_max:.2f}")
    print("\n模拟参数:")
    print(f"  f点数: {len(f_values)}")
    print(f"  r扫描点数: 200")
    print(f"  n扫描点数: 100")
    print("\n理论计算参数 (fsolve):")
    print(f"  r点数: {len(r_values)}")
    print(f"  初始猜测策略: 迭代使用前一个r值的结果作为初始猜测")
    
# ============ 运行主程序 ============
if __name__ == "__main__":
    main()
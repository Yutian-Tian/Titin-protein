"""
系统：由2个domain串联而成的链
目标：验证当能量惩罚DeltaE相同，折叠长度xi_f不同，domain的打开顺序
能垒与能量惩罚相关，但是2个domain相同
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.signal import argrelextrema

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
tick_fontsize = 30
legend_fontsize = 25
legend_title_fontsize = 30

# 线宽和尺寸
axes_linewidth = 2.5
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 8
ytick_major_size = 8
grid_linewidth = 1.2
grid_alpha = 0.4
lines_linewidth = 3.5
lines_markersize = 12

# 刻度方向
xtick_direction = 'in'
ytick_direction = 'in'
xtick_top = False
ytick_right = False

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

def create_output_directory():
    """创建输出目录"""
    output_dir = "/home/tyt/project/Single-chain/opt+R/Rand_xi/simulation_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# Step 1: 参数设置
xi_f1 = 10.0
xi_f2 = 15.0
k = 2.0  # 展开系数，假设为2
xi_u1 = k * xi_f1
xi_u2 = k * xi_f2
DeltaE = 10.0
U0 = 20.0  

print("参数设置:")
print(f"ξ_f1 = {xi_f1}, ξ_f2 = {xi_f2}")
print(f"k = {k}, ξ_u1 = {xi_u1}, ξ_u2 = {xi_u2}")
print(f"ΔE = {DeltaE}, U₀ = {U0}")
print()

# Step 2: 轮廓长度的两种计算方式
def Lc_method1(n):
    """方法一：假定domain 1先打开"""
    Lc_0 = xi_f1 + xi_f2
    if n <= 1:
        return Lc_0 + n * (xi_u1 - xi_f1)
    else:
        Lc_1 = Lc_0 + 1 * (xi_u1 - xi_f1)
        return Lc_1 + (n - 1) * (xi_u2 - xi_f2)

def Lc_method2(n):
    """方法二：假定domain 2先打开"""
    Lc_0 = xi_f1 + xi_f2
    if n <= 1:
        return Lc_0 + n * (xi_u2 - xi_f2)
    else:
        Lc_1 = Lc_0 + 1 * (xi_u2 - xi_f2)
        return Lc_1 + (n - 1) * (xi_u1 - xi_f1)

# WLC自由能函数
def F_WLC(x, Lc):
    """WLC自由能公式"""
    if abs(1 - x**2) < 1e-10:  # 避免除零
        return np.inf
    term1 = (np.pi**2 / (2 * Lc)) * (1 - x**2)
    term2 = (2 * Lc) / (np.pi * (1 - x**2))
    return term1 + term2

# U(n)项
def U_func(n):
    """U(n) = ΔEn - U0 cos(2πn)"""
    return DeltaE * n - U0 * np.cos(2 * np.pi * n)

# 总自由能函数
def F_total(r, n, Lc_func):
    """计算总自由能：F_c(r, n) = F_WLC(x, n) + U(n)"""
    Lc = Lc_func(n)
    if r > Lc:  # r不能超过轮廓长度
        return np.inf
    x = r / Lc
    return F_WLC(x, Lc) + U_func(n)

# Step 3: 分析两种方法
def analyze_method(Lc_func, method_name):
    """分析指定方法"""
    # 计算最大轮廓长度
    Lc_max = 0.95 * Lc_func(2)
    
    # 离散化r
    r_points = 500
    r_values = np.linspace(0, Lc_max, r_points)
    
    # 离散化n
    n_points = 400
    n_values = np.linspace(0, 2, n_points)
    
    # 存储结果
    F_min_values = []
    n_opt_values = []
    
    # 对每个r扫描n
    for r in r_values:
        F_values = []
        valid_n_values = []
        
        for n in n_values:
            Lc = Lc_func(n)
            if r <= Lc:  # 只考虑r ≤ Lc的情况
                F_val = F_total(r, n, Lc_func)
                F_values.append(F_val)
                valid_n_values.append(n)
        
        if F_values:  # 如果有有效值
            F_min = min(F_values)
            min_idx = np.argmin(F_values)
            n_opt = valid_n_values[min_idx]
            
            F_min_values.append(F_min)
            n_opt_values.append(n_opt)
        else:
            F_min_values.append(np.nan)
            n_opt_values.append(np.nan)
    
    return r_values, F_min_values, n_opt_values, Lc_max

# 分析两种方法
print("分析方法一: domain 1先打开")
r1, F_min1, n_opt1, Lc_max1 = analyze_method(Lc_method1, "Method 1")
print(f"最大轮廓长度 Lc_max(2) = {Lc_max1:.2f}")
print()

print("分析方法二: domain 2先打开")
r2, F_min2, n_opt2, Lc_max2 = analyze_method(Lc_method2, "Method 2")
print(f"最大轮廓长度 Lc_max(2) = {Lc_max2:.2f}")
print()

# 指定输出目录
output_dir = '/home/tyt/project/Single-chain/opt+R/Rand_xi/simulation_results'

# Step 4: 寻找自由能差值的极值点
print("寻找自由能差值的极值点:")
print("="*50)

# 计算自由能差值
F_min1_array = np.array(F_min1)
F_min2_array = np.array(F_min2)
F_diff = F_min1_array - F_min2_array

# 找到有效数据点（非NaN）
valid_indices = ~np.isnan(F_diff)
if np.any(valid_indices):
    F_diff_valid = F_diff[valid_indices]
    r1_valid = r1[valid_indices]
    n_opt1_valid = np.array(n_opt1)[valid_indices]
    n_opt2_valid = np.array(n_opt2)[valid_indices]
    
    # 寻找局部极大值点
    local_max_indices = argrelextrema(F_diff_valid, np.greater, order=10)[0]
    
    # 寻找局部极小值点
    local_min_indices = argrelextrema(F_diff_valid, np.less, order=10)[0]
    
    print(f"找到 {len(local_max_indices)} 个局部极大值点:")
    for idx in local_max_indices:
        r_val = r1_valid[idx]
        F_diff_val = F_diff_valid[idx]
        n_opt1_val = n_opt1_valid[idx]
        n_opt2_val = n_opt2_valid[idx]
        print(f"  r = {r_val:.2f}, ΔF = {F_diff_val:.4f}, n_opt1 = {n_opt1_val:.4f}, n_opt2 = {n_opt2_val:.4f}")
    
    print(f"\n找到 {len(local_min_indices)} 个局部极小值点:")
    for idx in local_min_indices:
        r_val = r1_valid[idx]
        F_diff_val = F_diff_valid[idx]
        n_opt1_val = n_opt1_valid[idx]
        n_opt2_val = n_opt2_valid[idx]
        print(f"  r = {r_val:.2f}, ΔF = {F_diff_val:.4f}, n_opt1 = {n_opt1_val:.4f}, n_opt2 = {n_opt2_val:.4f}")
    
    # 寻找全局极值点
    if len(F_diff_valid) > 0:
        global_max_idx = np.nanargmax(F_diff_valid)
        global_min_idx = np.nanargmin(F_diff_valid)
        
        print(f"\n全局极值点:")
        print(f"全局最大值: r = {r1_valid[global_max_idx]:.2f}, ΔF = {F_diff_valid[global_max_idx]:.4f}, "
              f"n_opt1 = {n_opt1_valid[global_max_idx]:.4f}, n_opt2 = {n_opt2_valid[global_max_idx]:.4f}")
        print(f"全局最小值: r = {r1_valid[global_min_idx]:.2f}, ΔF = {F_diff_valid[global_min_idx]:.4f}, "
              f"n_opt1 = {n_opt1_valid[global_min_idx]:.4f}, n_opt2 = {n_opt2_valid[global_min_idx]:.4f}")
else:
    print("警告: 自由能差值数组中所有值均为NaN")

print("="*50)
print()

# Step 5: 双纵轴可视化 - 自由能比较和差值
print("正在生成可视化图表...")

# 创建图形和主纵轴
fig, ax1 = plt.subplots(figsize=(12, 9))

# 绘制两种方法的F_min - r曲线（主纵轴）
line1 = ax1.plot(r1, F_min1, color='blue', linewidth=lines_linewidth, 
                 label='Case 1: domain 1 opens first')
line2 = ax1.plot(r2, F_min2, color='red', linewidth=lines_linewidth, linestyle='--',
                 label='Case 2: domain 2 opens first')

# 设置主纵轴标签
ax1.set_xlabel('End-to-end distance $r$', fontsize=label_fontsize)
ax1.set_ylabel('Minimum free energy $F_{min}$', fontsize=label_fontsize, color='black')
ax1.tick_params(axis='y', labelcolor='black', which='major', 
                width=ytick_major_width, size=ytick_major_size,
                labelsize=tick_fontsize)
ax1.tick_params(axis='x', which='major', 
                width=xtick_major_width, size=xtick_major_size,
                labelsize=tick_fontsize)

ax1.set_xlim(0.0, Lc_max1)

# 创建次纵轴
ax2 = ax1.twinx()

# 绘制自由能差值曲线（次纵轴）
line3 = ax2.plot(r1, F_diff, color='green', linewidth=lines_linewidth*0.8, linestyle='-.',
                 label='$\Delta F = F_1 - F_2$')

# 在差值曲线上标记极值点
if np.any(valid_indices):
    # 标记局部极大值点
    for idx in local_max_indices:
        r_val = r1_valid[idx]
        F_diff_val = F_diff_valid[idx]
        ax2.plot(r_val, F_diff_val, '^', color='purple', markersize=10, markeredgewidth=2, 
                 markeredgecolor='black', label='Local max' if idx == local_max_indices[0] else "")
    
    # 标记局部极小值点
    for idx in local_min_indices:
        r_val = r1_valid[idx]
        F_diff_val = F_diff_valid[idx]
        ax2.plot(r_val, F_diff_val, 'v', color='orange', markersize=10, markeredgewidth=2, 
                 markeredgecolor='black', label='Local min' if idx == local_min_indices[0] else "")
    
    # 标记全局极值点
    if len(F_diff_valid) > 0:
        ax2.plot(r1_valid[global_max_idx], F_diff_valid[global_max_idx], '*', color='gold', 
                 markersize=15, markeredgewidth=2, markeredgecolor='black', label='Global max')
        ax2.plot(r1_valid[global_min_idx], F_diff_valid[global_min_idx], '*', color='cyan', 
                 markersize=15, markeredgewidth=2, markeredgecolor='black', label='Global min')

# 设置次纵轴标签
ax2.set_ylabel('Free energy difference $\Delta F$', fontsize=label_fontsize, color='green')
ax2.tick_params(axis='y', labelcolor='green', which='major', 
                width=ytick_major_width, size=ytick_major_size,
                labelsize=tick_fontsize)

# 在差值曲线上添加零线参考
ax2.axhline(y=0, color='green', linestyle=':', alpha=0.6, linewidth=grid_linewidth)

# 设置标题
ax1.set_title('Free energy comparison with extremum points', fontsize=title_fontsize, pad=20)

# 合并图例
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]

# 添加极值点标记到图例（如果存在）
if np.any(valid_indices):
    # 只添加一次每个类型的标记
    labels.extend(['Local max', 'Local min', 'Global max', 'Global min'])
    from matplotlib.lines import Line2D
    lines.append(Line2D([0], [0], marker='^', color='w', markerfacecolor='purple', 
                        markeredgecolor='black', markersize=10, markeredgewidth=2))
    lines.append(Line2D([0], [0], marker='v', color='w', markerfacecolor='orange', 
                        markeredgecolor='black', markersize=10, markeredgewidth=2))
    lines.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                        markeredgecolor='black', markersize=15, markeredgewidth=2))
    lines.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='cyan', 
                        markeredgecolor='black', markersize=15, markeredgewidth=2))

ax1.legend(lines, labels, fontsize=legend_fontsize-2, loc='upper left', 
           framealpha=0.9, edgecolor='none')

# 设置网格
ax1.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)

# 设置坐标轴刻度参数
ax1.tick_params(axis='both', which='major', 
                direction=xtick_direction,
                top=xtick_top,
                right=ytick_right,
                bottom=True,
                left=True,
                width=xtick_major_width,
                length=xtick_major_size)

# 开启次刻度
ax1.minorticks_on()
ax1.tick_params(axis='both', which='minor',
                direction=xtick_direction,
                top=xtick_top,
                right=ytick_right,
                bottom=True,
                left=True,
                width=xtick_major_width*0.75,
                length=xtick_major_size*0.5)

# 强化边框
for spine in ax1.spines.values():
    spine.set_linewidth(axes_linewidth)

# 自动调整布局
plt.tight_layout()

# 保存图像
output_file = f"{output_dir}/free_energy_comparison_consenergy.png"
plt.savefig(output_file, dpi=savefig_dpi, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print(f"结果已保存到: {output_file}")

# Step 6: 添加新图 - 绘制n_opt-r曲线
print("\n正在生成n_opt-r曲线图...")

# 创建新的图形
fig2, ax3 = plt.subplots(figsize=(12, 9))

# 绘制两种方法的n_opt-r曲线
line4 = ax3.plot(r1, n_opt1, color='blue', linewidth=lines_linewidth, 
                 label='Case 1: domain 1 opens first')
line5 = ax3.plot(r2, n_opt2, color='red', linewidth=lines_linewidth, linestyle='--',
                 label='Case 2: domain 2 opens first')

# 设置坐标轴标签
ax3.set_xlabel('End-to-end distance $r$', fontsize=label_fontsize)
ax3.set_ylabel('Optimal unfolded domains $n_u$', fontsize=label_fontsize)
ax3.tick_params(axis='both', which='major', 
                width=xtick_major_width, size=xtick_major_size,
                labelsize=tick_fontsize)

# 设置x轴范围
ax3.set_xlim(0.0, 48.0)
# 设置y轴范围
ax3.set_ylim(-0.1, 2.1)
# 设置y轴刻度
ax3.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
# 添加水平参考线表示整数n值
ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.4, linewidth=grid_linewidth)
ax3.axhline(y=1, color='gray', linestyle=':', alpha=0.4, linewidth=grid_linewidth)
ax3.axhline(y=2, color='gray', linestyle=':', alpha=0.4, linewidth=grid_linewidth)

# 在n_opt曲线上标记自由能差值极值点对应的位置
if np.any(valid_indices):
    # 标记局部极大值点对应的n_opt
    for idx in local_max_indices:
        r_val = r1_valid[idx]
        n_opt1_val = n_opt1_valid[idx]
        n_opt2_val = n_opt2_valid[idx]
        ax3.plot(r_val, n_opt1_val, '^', color='purple', markersize=10, 
                 markeredgewidth=2, markeredgecolor='black')
        ax3.plot(r_val, n_opt2_val, '^', color='purple', markersize=10, 
                 markeredgewidth=2, markeredgecolor='black', alpha=0.5)
    
    # 标记局部极小值点对应的n_opt
    for idx in local_min_indices:
        r_val = r1_valid[idx]
        n_opt1_val = n_opt1_valid[idx]
        n_opt2_val = n_opt2_valid[idx]
        ax3.plot(r_val, n_opt1_val, 'v', color='orange', markersize=10, 
                 markeredgewidth=2, markeredgecolor='black')
        ax3.plot(r_val, n_opt2_val, 'v', color='orange', markersize=10, 
                 markeredgewidth=2, markeredgecolor='black', alpha=0.5)
    
    # 标记全局极值点对应的n_opt
    if len(F_diff_valid) > 0:
        ax3.plot(r1_valid[global_max_idx], n_opt1_valid[global_max_idx], '*', 
                 color='gold', markersize=15, markeredgewidth=2, markeredgecolor='black')
        ax3.plot(r1_valid[global_max_idx], n_opt2_valid[global_max_idx], '*', 
                 color='gold', markersize=15, markeredgewidth=2, markeredgecolor='black', alpha=0.5)
        ax3.plot(r1_valid[global_min_idx], n_opt1_valid[global_min_idx], '*', 
                 color='cyan', markersize=15, markeredgewidth=2, markeredgecolor='black')
        ax3.plot(r1_valid[global_min_idx], n_opt2_valid[global_min_idx], '*', 
                 color='cyan', markersize=15, markeredgewidth=2, markeredgecolor='black', alpha=0.5)

# 设置标题
ax3.set_title('$n_u$ vs. distance $r$ with extremum points', fontsize=title_fontsize, pad=20)

# 将图例放在左上角
ax3.legend(fontsize=legend_fontsize, loc='upper left', 
           framealpha=0.9, edgecolor='none')

# 设置网格
ax3.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)

# 设置坐标轴刻度参数
ax3.tick_params(axis='both', which='major', 
                direction=xtick_direction,
                top=xtick_top,
                right=ytick_right,
                bottom=True,
                left=True,
                width=xtick_major_width,
                length=xtick_major_size)

# 开启次刻度
ax3.minorticks_on()
ax3.tick_params(axis='both', which='minor',
                direction=xtick_direction,
                top=xtick_top,
                right=ytick_right,
                bottom=True,
                left=True,
                width=xtick_major_width*0.75,
                length=xtick_major_size*0.5)

# 强化边框
for spine in ax3.spines.values():
    spine.set_linewidth(axes_linewidth)

# 自动调整布局
plt.tight_layout()

# 保存图像
output_file2 = f"{output_dir}/nopt_vs_r_consenergy.png"
plt.savefig(output_file2, dpi=savefig_dpi, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print(f"结果已保存到: {output_file2}")

# 显示统计信息
print("\n统计信息:")
print(f"方法1: 当r从{min(r1):.1f}增加到{max(r1):.1f}时，n_opt从{n_opt1[0]:.2f}增加到{n_opt1[-1]:.2f}")
print(f"方法2: 当r从{min(r2):.1f}增加到{max(r2):.1f}时，n_opt从{n_opt2[0]:.2f}增加到{n_opt2[-1]:.2f}")

# 计算n_opt差值
n_diff = np.array(n_opt1) - np.array(n_opt2)
print(f"n_opt平均差值: {np.nanmean(n_diff):.4f}")
print(f"n_opt最大差值: {np.nanmax(np.abs(n_diff)):.4f}")

print("\n程序执行完成!")
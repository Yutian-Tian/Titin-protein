"""
目的：用于比较2 domian、4 domain和6 domain的本构曲线
输入：读取3组平均拉伸行为的数据
输出：可视化三组本构曲线
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.interpolate import interp1d
import sys

# ============ 字体设置 ============
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
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
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

# 基本参数
xi_f = 5.0  # 折叠态持续长度
alpha = 7.0      # 解折叠系数
E0 = 1.0     # 能量基准值
Ek = 4.0     # 能量系数
Nmax = 6.0     # domain 的数量
k1 = 10.0
k2 = 1.35
R0 = 5.0    # 初始首末端距离


def StressOptimization(R0, r_val, f_val):
    r_val = np.asarray(r_val)
    f_val = np.asarray(f_val)

    # 只保留 r >= R0 的数据点
    mask = r_val >= R0
    if not np.any(mask):
        raise ValueError("没有找到 r >= R0 的数据点，请检查 R0 或数据范围。")

    r_selected = r_val[mask]
    lambda_ = r_selected / R0                     # λ = r / R0
    r2 = lambda_ ** (-0.5) * R0                   # 对应的另一条链伸长

    # 线性插值获取力值，超出范围时使用边界值
    f1 = np.interp(r_selected, r_val, f_val, left=f_val[0], right=f_val[-1])
    f2 = np.interp(r2, r_val, f_val, left=f_val[0], right=f_val[-1])

    # 可选：检查外推警告
    if np.any(r_selected < r_val[0]) or np.any(r_selected > r_val[-1]):
        print("警告：某些 r1 值超出原始数据范围，使用了边界值。")
    if np.any(r2 < r_val[0]) or np.any(r2 > r_val[-1]):
        print("警告：某些 r2 值超出原始数据范围，使用了边界值。")

    # 计算应力：σ = R0 [ F'(λR0) - λ^{-3/2} F'(λ^{-1/2}R0) ]
    sigma = R0 * (f1 - lambda_ ** (-1.5) * f2)

    # 确保包含 λ=1 且 σ=0
    eps = 1e-12
    if len(lambda_) == 0:
        return lambda_, sigma   # 实际不会发生，因前面已检查非空

    # 检查第一个点（即最小 λ）是否接近 1
    if np.abs(lambda_[0] - 1.0) > eps:
        # 不包含 λ=1，插入 (1, 0) 到数组开头
        lambda_ = np.concatenate(([1.0], lambda_))
        sigma = np.concatenate(([0.0], sigma))
    else:
        # 已包含 λ≈1，将该点应力精确设为 0
        sigma[0] = 0.0

    return lambda_, sigma


def create_visualization(save_dir):
    """创建可视化图表"""


    # ============ 创建3组3-chain model的本构曲线：N=2, N=4, N=6============
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    file_path1 = os.path.join(save_dir,"2_100_C_file/average_curves.csv")
    file_path2 = os.path.join(save_dir,"4_100_C_file/average_curves.csv")
    file_path3 = os.path.join(save_dir,"6_100_C_file/average_curves.csv")

    #读取第1组数据
    df1 = pd.read_csv(file_path1, header=0)
    f1_vals = df1.iloc[:, 0].astype(float).values
    r1_vals = df1.iloc[:, 1].astype(float).values
    #读取第2组数据
    df2 = pd.read_csv(file_path2, header=0)
    f2_vals = df2.iloc[:, 0].astype(float).values
    r2_vals = df2.iloc[:, 1].astype(float).values
    #读取第3组数据
    df3 = pd.read_csv(file_path3, header=0)
    f3_vals = df3.iloc[:, 0].astype(float).values
    r3_vals = df3.iloc[:, 1].astype(float).values

    # 数值曲线：红色
    # 第1组
    valid_mask1 = ~np.isnan(f1_vals)
    if np.any(valid_mask1):
        lambda_1, sigma1 = StressOptimization(R0, r1_vals[valid_mask1], f1_vals[valid_mask1])
    ax.plot(lambda_1, sigma1, color='red', linewidth=lines_linewidth, label='N = 2', zorder=2)
    # 第2组
    valid_mask2 = ~np.isnan(f2_vals)
    if np.any(valid_mask2):
        lambda_2, sigma2 = StressOptimization(R0, r2_vals[valid_mask2], f2_vals[valid_mask2])
    ax.plot(lambda_2, sigma2, color='blue', linewidth=lines_linewidth, label='N = 4', zorder=2)
    # 第3组
    valid_mask3 = ~np.isnan(f3_vals)
    if np.any(valid_mask3):
        lambda_3, sigma3 = StressOptimization(R0, r3_vals[valid_mask3], f3_vals[valid_mask3])
    ax.plot(lambda_3, sigma3, color='purple', linewidth=lines_linewidth, label='N = 6', zorder=2)

    # 设置标签和标题
    ax.set_xlabel('Stretch ratio $\lambda$', fontsize=label_fontsize)
    ax.set_ylabel('Stress $\sigma/\\rho k_B T R_0$', fontsize=label_fontsize)
    ax.set_title(f'Constitutive curve of 3-chain model', 
                  fontsize=title_fontsize, pad=20)
    
    # 设置网格
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    
    # 设置图例
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, 
               edgecolor='none', loc='best')
    
    # 设置坐标轴范围
    ax.set_xlim(1.0, 2.0)
    ax.set_ylim(0.0, 5.0)
    
    # 设置刻度参数
    ax.tick_params(axis='both', which='major', 
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True,
                    left=True,
                    width=xtick_major_width,
                    length=xtick_major_size)
    
    ax.tick_params(axis='both', which='minor',
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True,
                    left=True,
                    width=xtick_major_width*0.75,
                    length=xtick_major_size*0.5)
    
    # 开启次刻度
    ax.minorticks_on()
    
    # 强化边框
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)
    
    plt.tight_layout()
    
    # 保存第三幅图
    if save_dir:
        save_path = os.path.join(save_dir, 'Stress-strain_Compare.png')
        fig.savefig(save_path, dpi=savefig_dpi, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        print(f"本构曲线已保存至: {save_path}")

    
    return fig

def main():

    save_dir = "/home/tyt/project/Single-chain/opt+R/Rand_xi/Gibbs_Optimization_results/100_chains_IMS"

    create_visualization(save_dir)

    print("Process completed!")

# ============ 运行主程序 ============
if __name__ == "__main__":
    main()
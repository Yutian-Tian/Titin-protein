#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化f-r曲线

功能：
1. 从 r1.csv, n1.csv, n2.csv 读取数据，每文件第一列为系统公共r值，其余列对应不同实验组。
2. 对每组数据计算系统力（两个串联域的WLC模型，假设 r2 = r - r1，取两域力的平均值）。
3. 直接使用公共r作为横坐标，计算所有组的平均力曲线（忽略NaN）。
4. 绘制平均力曲线（红色）和所有原始轨迹（半透明灰色细线），样式与代码2一致。
5. 添加力平衡检测：输出每个数据点两域力差的统计信息（平均绝对差、最大绝对差、标准差）。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.interpolate import interp1d  # 本版本不再需要插值，但保留以备将来

# ============ 用户需修改的路径 ============
# 请根据实际情况修改以下三个文件的路径
path = "/home/tyt/project/Single-chain/opt+R/Rand_xi/Helmholtz_Optimization_results/100chains"
r1_file = os.path.join(path, "r1.csv")
n1_file = os.path.join(path, "n1.csv")
n2_file = os.path.join(path, "n2.csv")

# 输出图像保存目录（若为None则只显示不保存）
output_dir = "/home/tyt/project/Single-chain/opt+R/Rand_xi/Helmholtz_Optimization_results/100chains"
# 自动生成文件名
output_file = os.path.join(output_dir, "fr_curve.png") if output_dir else None

# ============ 物理参数（与代码1一致） ============
xi_f = 5.0          # 折叠态长度（两个域相同）
k = 1.1             # k = xi_ui/xi_fi
f_limit = 25.0        # 力显示上限

# ============ 字体路径（与代码2一致） ============
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

# ============ 样式变量定义（与代码2完全一致） ============
font_family = 'Times New Roman'
font_weight = 'normal'
math_fontset = 'stix'
math_rm = 'Times New Roman'
math_it = 'Times New Roman:italic'
math_bf = 'Times New Roman:bold'

title_fontsize = 35
label_fontsize = 35
tick_fontsize = 35
legend_fontsize = 25
legend_title_fontsize = 35

axes_linewidth = 2
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 10
ytick_major_size = 10
grid_linewidth = 1
grid_alpha = 0.4
lines_linewidth = 5
lines_markersize = 15

xtick_direction = 'in'
ytick_direction = 'in'
xtick_top = True
ytick_right = True

figure_dpi = 100
savefig_dpi = 300

# ============ 应用全局样式 ============
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

# ============ 力计算函数（与代码1一致） ============
def contour_length_Lci(n_i, xi_fi):
    """
    轮廓长度: L_{ci}(n_i) = ξ_fi + n_i (k - 1)ξ_fi
    """
    n_i = np.asarray(n_i, dtype=float)
    return xi_fi + n_i * (k - 1) * xi_fi

def MS_force(r, L_c):
    """
    Marko–Siggia 力（WLC模型近似）
    """
    x = np.asarray(r, dtype=float) / np.asarray(L_c, dtype=float)
    force = np.where(x < 1.0,
                     0.25 * ((1 - x) ** (-2) - 1 + 4 * x),
                     np.inf)
    return force

# ============ 修改后的系统力计算函数（增加力差输出） ============
def compute_system_force_with_diff(r, r1, n1, n2):
    """
    计算系统力（两个串联域的平均力），同时返回两个域的力差 force1 - force2
    参数：
        r: 系统总延伸（标量或数组）
        r1: 第一个域的延伸
        n1, n2: 两个域的折叠数
    返回：
        system_force, force_diff
    """
    # 计算第二个域的延伸
    r2 = r - r1
    # 确保延伸非负
    r2 = np.maximum(r2, 0.0)

    # 轮廓长度
    Lc1 = contour_length_Lci(n1, xi_f)
    Lc2 = contour_length_Lci(n2, xi_f)

    # 各域力
    force1 = MS_force(r1, Lc1)
    force2 = MS_force(r2, Lc2)

    # 无穷大力替换为NaN
    force1 = np.where(np.isfinite(force1), force1, np.nan)
    force2 = np.where(np.isfinite(force2), force2, np.nan)

    # 系统力取两个域力的平均值（忽略NaN）
    with np.errstate(invalid='ignore'):
        system_force = np.nanmean([force1, force2], axis=0)

    # 力差（如果任一力为NaN，差也为NaN）
    force_diff = force1 - force2
    force_diff = np.where(np.isnan(force1) | np.isnan(force2), np.nan, force_diff)

    return system_force, force_diff

# ============ 数据处理主流程（修改后） ============
def load_and_process_data():
    """
    加载三个CSV文件，计算每组系统力，返回公共r网格和所有组的力数据，并统计力平衡情况
    """
    print("=" * 80)
    print("开始读取数据...")
    # 读取文件，跳过第一行（标签）
    try:
        r1_df = pd.read_csv(r1_file, header=0)
        n1_df = pd.read_csv(n1_file, header=0)
        n2_df = pd.read_csv(n2_file, header=0)
    except Exception as e:
        print(f"读取文件出错: {e}")
        return None, None, None

    # 检查列数是否一致
    if not (r1_df.shape[1] == n1_df.shape[1] == n2_df.shape[1]):
        print("警告：三个文件的列数不一致，将使用最小列数")
        min_cols = min(r1_df.shape[1], n1_df.shape[1], n2_df.shape[1])
        r1_df = r1_df.iloc[:, :min_cols]
        n1_df = n1_df.iloc[:, :min_cols]
        n2_df = n2_df.iloc[:, :min_cols]

    # 公共r值（取第一个文件的第一列，假设三个文件的第一列相同）
    r_common = r1_df.iloc[:, 0].values.astype(float)
    # 组数 = 列数 - 1（减去第一列）
    num_groups = r1_df.shape[1] - 1
    print(f"成功读取数据，共 {num_groups} 组，公共r长度 {len(r_common)}")

    # 存储所有组的力数据
    all_forces = []
    # 存储力差统计（每组每个点的力差）
    force_diffs = []

    for i in range(num_groups):
        # 列索引：第0列是公共r，第i+1列是对应组的数据
        r1_col = r1_df.iloc[:, i+1].values.astype(float)
        n1_col = n1_df.iloc[:, i+1].values.astype(float)
        n2_col = n2_df.iloc[:, i+1].values.astype(float)

        # 计算系统力和力差
        force, diff = compute_system_force_with_diff(r_common, r1_col, n1_col, n2_col)
        all_forces.append(force)
        force_diffs.append(diff)

    # 转换为numpy数组
    all_forces = np.array(all_forces)
    force_diffs = np.array(force_diffs)  # shape (num_groups, len(r_common))

    # 统计力差（忽略NaN）
    valid_diffs = force_diffs[~np.isnan(force_diffs)]
    if len(valid_diffs) > 0:
        mean_abs_diff = np.mean(np.abs(valid_diffs))
        max_abs_diff = np.max(np.abs(valid_diffs))
        std_diff = np.std(valid_diffs)
        print("\n力平衡检测结果（两个域力的差值 force1 - force2）：")
        print(f"  平均绝对差: {mean_abs_diff:.6f}")
        print(f"  最大绝对差: {max_abs_diff:.6f}")
        print(f"  标准差: {std_diff:.6f}")
        if max_abs_diff > 1e-6:
            print("  注意：存在较大的力差，模型可能不满足严格的力平衡条件。")
        else:
            print("  力差在数值精度内，满足力平衡。")
    else:
        print("无法计算力差（所有点均为NaN）。")

    # 计算平均值（忽略NaN）
    f_avg = np.nanmean(all_forces, axis=0)

    # 输出统计
    valid_groups = np.sum(~np.isnan(all_forces).all(axis=1))  # 至少有一个非NaN的组
    print(f"\n有效组数（至少有部分有效数据）: {valid_groups}")

    return r_common, f_avg, all_forces

def visualize(r_common, f_avg, all_curves, save_path=None):
    """
    绘制平均曲线（红色）和所有原始曲线（半透明灰细线）
    """
    print("\n开始绘图...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    # 绘制所有原始曲线
    for curve in all_curves:
        ax.plot(r_common, curve, color='gray', alpha=0.3, linewidth=0.8, zorder=1)

    # 绘制平均曲线
    ax.plot(r_common, f_avg, color='red', linewidth=lines_linewidth,
            label=f'Average of {len(all_curves)} curves', zorder=3)

    # 坐标轴标签和标题
    ax.set_xlabel('End-to-end distance $r$', fontsize=label_fontsize)
    ax.set_ylabel('Force $f$', fontsize=label_fontsize)
    ax.set_title('Force vs. distance', fontsize=title_fontsize, pad=20)

    # 网格
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)

    # 图例
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')

    # 坐标轴范围
    ax.set_xlim(0, np.max(r_common) * 0.95)
    ax.set_ylim(0, f_limit)  # 从0开始，可根据数据调整

    # 刻度设置
    ax.tick_params(axis='both', which='major',
                   direction=xtick_direction,
                   top=xtick_top, right=ytick_right,
                   bottom=True, left=True,
                   width=xtick_major_width, length=xtick_major_size)
    ax.tick_params(axis='both', which='minor',
                   direction=xtick_direction,
                   top=xtick_top, right=ytick_right,
                   bottom=True, left=True,
                   width=xtick_major_width*0.75, length=xtick_major_size*0.5)
    ax.minorticks_on()

    # 边框宽度
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=savefig_dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"图形已保存至: {save_path}")
    else:
        plt.show()

    plt.close(fig)

# ============ 主程序 ============
def main():
    print("=" * 80)
    print("f-r曲线可视化程序（含力平衡检测）")
    print("=" * 80)

    # 检查文件是否存在
    for fpath in [r1_file, n1_file, n2_file]:
        if not os.path.exists(fpath):
            print(f"错误：文件不存在 - {fpath}")
            return

    # 处理数据
    r_common, f_avg, all_curves = load_and_process_data()
    if r_common is None:
        print("数据处理失败，程序退出")
        return

    # 可视化
    visualize(r_common, f_avg, all_curves, output_file)

    # 输出简单统计
    print("\n统计信息:")
    print(f"  有效组数: {len(all_curves)}")
    print(f"  公共r点数: {len(r_common)}")
    print(f"  平均曲线有效点数: {np.sum(~np.isnan(f_avg))}")
    print(f"  平均力范围: {np.nanmin(f_avg):.4f} 到 {np.nanmax(f_avg):.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
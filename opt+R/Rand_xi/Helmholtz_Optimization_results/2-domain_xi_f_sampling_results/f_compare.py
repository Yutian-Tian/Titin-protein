#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理代码一（独立网格）生成的数据：
- 从 xi_f.csv 读取每条链的折叠长度 xi_f1, xi_f2
- 从 r1_values.csv, r2_values.csv, n1_values.csv, n2_values.csv 读取优化结果
- 计算每个链在各自拉伸状态下的系统力（两域WLC力的平均）
- 插值到公共拉伸网格
- 绘制平均力曲线和所有原始曲线（半透明灰色细线）
- 输出力平衡检测（两域力差的统计信息）
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.interpolate import interp1d

# ==================== 用户配置（请修改为实际路径） ====================
# 代码一输出目录
data_dir = "/home/tyt/project/Single-chain/opt+R/Rand_xi/Helmholtz_Optimization_results/2-domain_xi_f_sampling_results"

# 输出图像保存目录（若为None则只显示不保存）
output_dir = os.path.join(data_dir, "Figure")   # 可以改为其他目录
output_file = os.path.join(output_dir, "f_r_curve.png") if output_dir else None

# ==================== 物理参数（必须与代码一一致） ====================
alpha = 4.0          # xi_u = alpha * xi_f  (代码一中的 alpha)
r_grids = 1000       # 每条链原始网格点数（代码一中的 r_grids）
common_grid_points = 1000   # 公共网格点数（可自行调整）

# 力显示上限（可根据数据调整）
f_limit = 10.0

# ==================== 字体与绘图样式（与代码二完全一致） ====================
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

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

# 应用全局样式
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

# ==================== 物理函数 ====================
def contour_length_Lci(n_i, xi_fi):
    """轮廓长度: L_ci = xi_fi + n_i * (alpha-1) * xi_fi"""
    n_i = np.asarray(n_i, dtype=float)
    return xi_fi + n_i * (alpha - 1) * xi_fi

def MS_force(r, L_c):
    """Marko–Siggia 力 (WLC模型)"""
    x = np.asarray(r, dtype=float) / np.asarray(L_c, dtype=float)
    force = np.where(x < 1.0,
                     0.25 * ((1 - x) ** (-2) - 1 + 4 * x),
                     np.inf)
    return force

def compute_force_for_chain(r1, r2, n1, n2, xi_f1, xi_f2):
    """
    计算单条链在各拉伸点上的系统力（两域力的平均值）
    同时返回两域力差 force1 - force2（用于力平衡检测）
    """
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)
    n1 = np.asarray(n1)
    n2 = np.asarray(n2)


    Lc1 = contour_length_Lci(n1, xi_f1)
    Lc2 = contour_length_Lci(n2, xi_f2)

    force1 = MS_force(r1, Lc1)
    force2 = MS_force(r2, Lc2)

    # 无穷大替换为NaN
    force1 = np.where(np.isfinite(force1), force1, np.nan)
    force2 = np.where(np.isfinite(force2), force2, np.nan)

    # 系统力 = 平均值
    system_force = (force1 + force2) / 2.0

    # 力差（仅当两力均有效时）
    force_diff = force1 - force2
    force_diff = np.where(np.isnan(force1) | np.isnan(force2), np.nan, force_diff)

    return system_force, force_diff

# ==================== 主程序 ====================
def main():
    print("=" * 80)
    print("处理代码一生成的独立网格数据")
    print("=" * 80)

    # ---------- 1. 读取 xi_f.csv 获取每条链的 xi_f1, xi_f2 ----------
    xi_f_path = os.path.join(data_dir, "xi_f.csv")
    if not os.path.exists(xi_f_path):
        raise FileNotFoundError(f"未找到文件: {xi_f_path}")
    df_xi = pd.read_csv(xi_f_path)
    # 期望列: group, xi_f1, xi_f2
    # 创建映射: chain_index (从0开始) -> (xi_f1, xi_f2)
    xi_f_dict = {}
    for idx, row in df_xi.iterrows():
        group = int(row['group'])
        xi_f_dict[group - 1] = (row['xi_f1'], row['xi_f2'])
    num_chains = len(xi_f_dict)
    print(f"读取到 {num_chains} 条链的 xi_f 数据")

    # ---------- 2. 读取 r1, r2, n1, n2 数据文件 ----------
    def load_values_file(filename):
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"未找到文件: {filepath}")
        df = pd.read_csv(filepath)
        # 列名应为 chain_1, chain_2, ...  (索引从1开始)
        return df

    r1_df = load_values_file("r1_values.csv")
    r2_df = load_values_file("r2_values.csv")
    n1_df = load_values_file("n1_values.csv")
    n2_df = load_values_file("n2_values.csv")

    # 检查列数一致性
    assert r1_df.shape[1] == r2_df.shape[1] == n1_df.shape[1] == n2_df.shape[1] == num_chains
    print(f"数据文件读取成功，每条链原始网格点数 = {r1_df.shape[0]}")

    # ---------- 3. 为每条链重建实际拉伸长度 r = r1 + r2 ----------
    r_original = []          # 存储每条链的原始拉伸长度数组
    force_original = []      # 存储每条链的原始系统力数组
    force_diff_original = [] # 存储每条链的原始力差数组

    for i in range(num_chains):
        col_name = f'chain_{i+1}'
        r1 = r1_df[col_name].values.astype(float)
        r2 = r2_df[col_name].values.astype(float)
        n1 = n1_df[col_name].values.astype(float)
        n2 = n2_df[col_name].values.astype(float)

        r = r1 + r2

        xi_f1, xi_f2 = xi_f_dict[i]

        force, diff = compute_force_for_chain(r1, r2, n1, n2, xi_f1, xi_f2)

        r_original.append(r)
        force_original.append(force)
        force_diff_original.append(diff)

    # ---------- 4. 确定公共拉伸网格 ----------
    # 找出所有链的最大有效拉伸长度（忽略NaN点对应的r）
    all_r_max = []
    for r_arr in r_original:
        # 去除NaN对应点（但r_arr本身没有NaN，是有限值）
        valid_r = r_arr[np.isfinite(r_arr)]
        if len(valid_r) > 0:
            all_r_max.append(np.max(valid_r))
    global_r_max = 0.95 * np.max(all_r_max)  # 略小于理论最大，避免边界奇异
    r_common = np.linspace(0, global_r_max, common_grid_points)
    print(f"公共网格范围: [0, {global_r_max:.4f}], 点数: {common_grid_points}")

    # ---------- 5. 插值每条链的力曲线到公共网格 ----------
    forces_common = np.full((common_grid_points, num_chains), np.nan)  # 行=公共点，列=链

    for i in range(num_chains):
        r_i = r_original[i]
        f_i = force_original[i]

        # 去除无效点（NaN或inf）
        valid_mask = np.isfinite(r_i) & np.isfinite(f_i)
        if not np.any(valid_mask):
            print(f"警告: 链 {i+1} 没有有效力数据，跳过插值")
            continue

        r_valid = r_i[valid_mask]
        f_valid = f_i[valid_mask]

        # 确保r单调递增（原始数据是均匀网格，天然递增）
        # 插值函数，外推填NaN
        try:
            interp_func = interp1d(r_valid, f_valid, kind='linear',
                                   bounds_error=False, fill_value=np.nan)
            f_interp = interp_func(r_common)
            forces_common[:, i] = f_interp
        except Exception as e:
            print(f"插值链 {i+1} 时出错: {e}")

    # ---------- 6. 计算平均力曲线 ----------
    f_mean = np.nanmean(forces_common, axis=1)
    # 可选：计算标准差
    f_std = np.nanstd(forces_common, axis=1)

    # ---------- 7. 力平衡检测：统计所有链两域力差的绝对值 ----------
    all_diffs = []
    for i in range(num_chains):
        diff_i = force_diff_original[i]
        valid_diff = diff_i[np.isfinite(diff_i)]
        all_diffs.extend(valid_diff)

    if len(all_diffs) > 0:
        mean_abs_diff = np.mean(np.abs(all_diffs))
        max_abs_diff = np.max(np.abs(all_diffs))
        max_idx = np.argmax(np.abs(all_diffs))
        std_diff = np.std(all_diffs)
        print("\n力平衡检测结果（两域力差 force1 - force2）：")
        print(f"  平均绝对差: {mean_abs_diff:.6e}")
        print(f"  最大绝对差: {max_abs_diff:.6e}")
        print(f"  标准差: {std_diff:.6e}")
        if max_abs_diff > 1e-6:
            print(f"  注意：第{max_idx}组数据存在较大的力差，优化可能未严格满足力平衡。")
        else:
            print("  力差在数值精度内，满足力平衡。")
    else:
        print("无法计算力差（无有效数据点）。")

    # ---------- 8. 绘图 ----------
    print("\n开始绘图...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    # 绘制所有原始曲线（半透明灰色细线）
    for i in range(num_chains):
        # 注意：这里应绘制插值后的公共网格曲线，否则横坐标不一致无法一起绘制
        # 为了显示原始曲线的形状，使用插值后的曲线（公共横坐标）
        f_chain = forces_common[:, i]
        # 只绘制有效部分
        mask = np.isfinite(f_chain)
        if np.any(mask):
            ax.plot(r_common[mask], f_chain[mask],
                    color='gray', alpha=0.3, linewidth=0.8, zorder=1)

    # 绘制平均曲线
    mask_mean = np.isfinite(f_mean)
    ax.plot(r_common[mask_mean], f_mean[mask_mean],
            color='red', linewidth=lines_linewidth,
            label=f'Average of {num_chains} chains', zorder=3)

    # 坐标轴标签和标题
    ax.set_xlabel('End-to-end distance $r$', fontsize=label_fontsize)
    ax.set_ylabel('Force $f$', fontsize=label_fontsize)
    ax.set_title('Force vs. distance', fontsize=title_fontsize, pad=20)

    # 网格
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)

    # 图例
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')

    # 坐标轴范围
    ax.set_xlim(0, global_r_max)
    ax.set_ylim(0, f_limit)

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

    if output_file:
        save_dir = os.path.dirname(output_file)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(output_file, dpi=savefig_dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"图形已保存至: {output_file}")
    else:
        plt.show()

    plt.close(fig)

    # 输出简单统计
    print("\n统计信息:")
    print(f"  链数量: {num_chains}")
    print(f"  公共网格点数: {common_grid_points}")
    print(f"  平均曲线有效点数: {np.sum(~np.isnan(f_mean))}")
    print(f"  平均力范围: {np.nanmin(f_mean):.4f} 到 {np.nanmax(f_mean):.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
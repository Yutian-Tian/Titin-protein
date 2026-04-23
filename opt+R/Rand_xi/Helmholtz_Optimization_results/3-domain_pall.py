"""
目的：探究不同参数下的三个domain的解折叠顺序，参数包括xi_f1, beta2, beta3, E1, E2, E3
系统：包含3个domain
主题：约束下优化系统的Helmholtz自由能
新增：可视化部分，绘制n1,n2,n3和力曲线，风格与参考代码一致
并行化：对r值进行多进程并行优化，大幅提高计算速度
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import minimize, Bounds
from multiprocessing import Pool, cpu_count

# ============ 字体路径（如有需要可修改） ============
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

# ============ 样式变量定义 ============
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
lines_linewidth = 1
lines_markersize = 15

xtick_direction = 'in'
ytick_direction = 'in'
xtick_top = True
ytick_right = True

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

# ============ 物理参数 ============
xi_f1 = 5.0          # 第一个domain的折叠态长度
k = 2.0              # k = xi_ui/xi_fi
beta2 = 2.0          # xi_f2 / xi_f1
beta3 = 2.5          # xi_f3 / xi_f1
force_limit = 20.0   # 力曲线y轴上限
E0 = 2.0
delta1 = 0.0         # ΔE1 = E0 + delta1
delta2 = 0.1         # ΔE2 = E0 + delta2
delta3 = 0.2         # ΔE3 = E0 + delta3

# 优化参数
r_grids = 1000

# 设置存储路径
save_path = "/home/tyt/project/Single-chain/opt+R/Rand_xi/Helmholtz_Optimization_results/3_domain_results_pall"
os.makedirs(save_path, exist_ok=True)

# ---------- 辅助函数（优化部分） ----------
def energy_term_U(n_i, DeltaEi):
    """
    能量项: U(n_i) = ΔE_i n_i - ΔE_i cos(2π n_i)
    """
    return DeltaEi * n_i - DeltaEi * np.cos(2 * np.pi * n_i)

def contour_length_Lci(n_i, xi_fi):
    """
    轮廓长度: L_{ci}(n_i) = ξ_fi + n_i (k - 1)ξ_fi
    """
    return xi_fi + n_i * (k - 1) * xi_fi

def WLC_free_energy(x_i, L_ci):
    if x_i >= 1.0 - 1e-12:
        return 1e300
    return 0.25 * L_ci * (x_i**2 * (3.0 - 2.0 * x_i) / (1.0 - x_i))

def single_domain_free_energy(r_i, n_i, DeltaEi, xi_fi):
    L_ci = contour_length_Lci(n_i, xi_fi)
    if r_i < 0 or r_i > L_ci:
        return 1e300
    x_i = r_i / L_ci
    F_wlc = WLC_free_energy(x_i, L_ci)
    Ui = energy_term_U(n_i, DeltaEi)
    return F_wlc + Ui

def free_energy_3_domain(r, r1, r2, n1, n2, n3, DeltaE1, DeltaE2, DeltaE3, xi_f1, xi_f2, xi_f3):
    r3 = r - r1 - r2
    if r3 < 0 or r3 > contour_length_Lci(n3, xi_f3):
        return 1e300
    energy1 = single_domain_free_energy(r1, n1, DeltaE1, xi_f1)
    energy2 = single_domain_free_energy(r2, n2, DeltaE2, xi_f2)
    energy3 = single_domain_free_energy(r3, n3, DeltaE3, xi_f3)
    total = energy1 + energy2 + energy3
    if not np.isfinite(total):
        return 1e300
    return total

def Optimize_single_point(r, init_points=9, refine_levels=6, refine_points=9, tol=1e-6):
    """
    使用粗网格 + 局部细化优化给定拉伸 r 下的自由能。
    返回 [r, r1_opt, r2_opt, n1_opt, n2_opt, n3_opt]
    """
    # 固定参数
    DeltaE1 = E0 + delta1
    DeltaE2 = E0 + delta2
    DeltaE3 = E0 + delta3
    xi_f2 = beta2 * xi_f1
    xi_f3 = beta3 * xi_f1

    n1_min, n1_max = 0.0, 1.0
    n2_min, n2_max = 0.0, 1.0
    n3_min, n3_max = 0.0, 1.0

    best_r1 = None
    best_r2 = None
    best_n1 = None
    best_n2 = None
    best_n3 = None
    best_F = float('inf')

    for level in range(refine_levels + 1):
        # 当前层网格点数
        N = init_points if level == 0 else refine_points
        n1_grid = np.linspace(n1_min, n1_max, N)
        n2_grid = np.linspace(n2_min, n2_max, N)
        n3_grid = np.linspace(n3_min, n3_max, N)

        level_best_F = float('inf')
        level_best_r1 = None
        level_best_r2 = None
        level_best_n1 = None
        level_best_n2 = None
        level_best_n3 = None

        # 遍历当前网格
        for n1 in n1_grid:
            Lc1 = contour_length_Lci(n1, xi_f1)
            for n2 in n2_grid:
                Lc2 = contour_length_Lci(n2, xi_f2)
                for n3 in n3_grid:
                    Lc3 = contour_length_Lci(n3, xi_f3)

                    # 可行性检查：总拉伸不能超过三个域最大轮廓长度之和，也不能小于0
                    if r < 0 or r > Lc1 + Lc2 + Lc3:
                        continue

                    # 定义目标函数（优化 r1, r2）
                    def obj(x):
                        r1v, r2v = x
                        return free_energy_3_domain(r, r1v, r2v, n1, n2, n3,
                                                    DeltaE1, DeltaE2, DeltaE3,
                                                    xi_f1, xi_f2, xi_f3)

                    # 边界: r1 ∈ [0, Lc1], r2 ∈ [0, Lc2]
                    bounds = Bounds([0.0, 0.0], [Lc1, Lc2])
                    # 线性约束: r1 + r2 >= r - Lc3  且  r1 + r2 <= r
                    cons = [
                        {'type': 'ineq', 'fun': lambda x: x[0] + x[1] - (r - Lc3)},  # >=0
                        {'type': 'ineq', 'fun': lambda x: r - (x[0] + x[1])}         # >=0
                    ]
                    # 初始猜测：均分
                    x0 = [min(Lc1, max(0.0, r/3)), min(Lc2, max(0.0, r/3))]

                    try:
                        res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons,
                                       options={'ftol': 1e-9, 'disp': False})
                        if res.success and res.fun < level_best_F:
                            level_best_F = res.fun
                            level_best_r1 = res.x[0]
                            level_best_r2 = res.x[1]
                            level_best_n1 = n1
                            level_best_n2 = n2
                            level_best_n3 = n3
                    except:
                        continue

        # 如果当前层无可行点，返回 NaN
        if level_best_F == float('inf'):
            return np.array([r, np.nan, np.nan, np.nan, np.nan, np.nan])

        # 更新全局最优
        if level_best_F < best_F:
            best_F = level_best_F
            best_r1 = level_best_r1
            best_r2 = level_best_r2
            best_n1 = level_best_n1
            best_n2 = level_best_n2
            best_n3 = level_best_n3

        # 最后一级细化结束
        if level == refine_levels:
            break

        # 确定下一级细化的范围：以当前最优为中心，取相邻网格点间距
        idx_n1 = np.argmin(np.abs(n1_grid - best_n1))
        idx_n2 = np.argmin(np.abs(n2_grid - best_n2))
        idx_n3 = np.argmin(np.abs(n3_grid - best_n3))

        left_n1 = n1_grid[max(0, idx_n1-1)]
        right_n1 = n1_grid[min(N-1, idx_n1+1)]
        left_n2 = n2_grid[max(0, idx_n2-1)]
        right_n2 = n2_grid[min(N-1, idx_n2+1)]
        left_n3 = n3_grid[max(0, idx_n3-1)]
        right_n3 = n3_grid[min(N-1, idx_n3+1)]

        n1_min, n1_max = left_n1, right_n1
        n2_min, n2_max = left_n2, right_n2
        n3_min, n3_max = left_n3, right_n3

        # 检查是否达到精度
        if (n1_max - n1_min < tol) and (n2_max - n2_min < tol) and (n3_max - n3_min < tol):
            break

    return np.array([r, best_r1, best_r2, best_n1, best_n2, best_n3])

# ---------- 并行处理包装函数 ----------
def process_single_r(r):
    """供多进程调用的函数，返回优化结果列表"""
    return Optimize_single_point(r).tolist()

# ---------- 可视化函数 ----------
def MS_force(r, L_c):
    """
    Marko–Siggia 力（WLC模型近似）
    """
    x = np.asarray(r, dtype=float) / np.asarray(L_c, dtype=float)
    force = np.where(x < 1.0,
                     0.25 * ((1 - x) ** (-2) - 1 + 4 * x),
                     1e15)
    return force

def plot_n_curves(ax, r, n1, n2, n3, title):
    """绘制 n1, n2, n3 vs r 曲线"""
    ax.plot(r, n1, color='blue', linewidth=lines_linewidth, label='$n_1$', zorder=3)
    ax.plot(r, n2, color='red', linewidth=lines_linewidth, linestyle='--', label='$n_2$', zorder=3)
    ax.plot(r, n3, color='green', linewidth=lines_linewidth, linestyle='-.', label='$n_3$', zorder=3)
    ax.set_xlabel('$r$', fontsize=label_fontsize)
    ax.set_ylabel('$n$', fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize, pad=20)
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    ax.tick_params(axis='both', which='major',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width, length=xtick_major_size)
    ax.tick_params(axis='both', which='minor',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width*0.75, length=xtick_major_size*0.5)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

def plot_force_curves(ax, r, force1, force2, force3, title):
    """绘制 f1, f2, f3 vs r 曲线"""
    ax.plot(r, force1, color='blue', linewidth=lines_linewidth, label='$f_1$', zorder=3)
    ax.plot(r, force2, color='red', linewidth=lines_linewidth, linestyle='--', label='$f_2$', zorder=3)
    ax.plot(r, force3, color='green', linewidth=lines_linewidth, linestyle='-.', label='$f_3$', zorder=3)
    ax.set_xlabel('$r$', fontsize=label_fontsize)
    ax.set_ylabel('$f$', fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize, pad=20)
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    ax.tick_params(axis='both', which='major',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width, length=xtick_major_size)
    ax.tick_params(axis='both', which='minor',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width*0.75, length=xtick_major_size*0.5)
    ax.minorticks_on()
    ax.set_xlim(0, r[-1])
    ax.set_ylim(0, force_limit)
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

# ---------- 主程序 ----------
def main():
    # 计算拉伸的最大可能值（三个域完全伸展时的总轮廓长度）
    xi_f2 = beta2 * xi_f1
    xi_f3 = beta3 * xi_f1
    r_max = (contour_length_Lci(1, xi_f1) +
             contour_length_Lci(1, xi_f2) +
             contour_length_Lci(1, xi_f3))
    r_vals = np.linspace(0, 0.95 * r_max, r_grids)

    # 并行处理
    num_workers = cpu_count()
    print(f"使用 {num_workers} 个进程并行计算...")

    results = []
    with Pool(processes=num_workers) as pool:
        # 使用 imap 保持结果顺序，并手动打印进度
        for i, res in enumerate(pool.imap(process_single_r, r_vals)):
            results.append(res)
            if (i + 1) % 100 == 0 or (i + 1) == len(r_vals):
                print(f"已处理 {i + 1} / {len(r_vals)} 个点")

    # 将结果转换为数组并提取各列
    results = np.array(results)
    r_opt = results[:, 0]
    r1_opt = results[:, 1]
    r2_opt = results[:, 2]
    n1_opt = results[:, 3]
    n2_opt = results[:, 4]
    n3_opt = results[:, 5]

    # 计算 r3
    r3_opt = r_opt - r1_opt - r2_opt
    r3_opt = np.where(np.isnan(r1_opt), np.nan, r3_opt)

    # 构建 DataFrame
    df = pd.DataFrame({
        'r': r_opt,
        'r1': r1_opt,
        'r2': r2_opt,
        'r3': r3_opt,
        'n1': n1_opt,
        'n2': n2_opt,
        'n3': n3_opt
    })
    csv_filename = os.path.join(save_path, "3_domain_results.csv")
    df.to_csv(csv_filename, index=False)
    print(f"结果已保存至: {csv_filename}")

    # ===== 可视化 =====
    r = df['r'].values
    r1 = df['r1'].values
    r2 = df['r2'].values
    r3 = df['r3'].values
    n1 = df['n1'].values
    n2 = df['n2'].values
    n3 = df['n3'].values

    # 计算轮廓长度和力
    xi_f2 = beta2 * xi_f1
    xi_f3 = beta3 * xi_f1
    Lc1 = contour_length_Lci(n1, xi_f1)
    Lc2 = contour_length_Lci(n2, xi_f2)
    Lc3 = contour_length_Lci(n3, xi_f3)
    force1 = MS_force(r1, Lc1)
    force2 = MS_force(r2, Lc2)
    force3 = MS_force(r3, Lc3)

    # 将无穷大力替换为 NaN
    force1 = np.where(np.isfinite(force1), force1, np.nan)
    force2 = np.where(np.isfinite(force2), force2, np.nan)
    force3 = np.where(np.isfinite(force3), force3, np.nan)

    # 标题
    title = (f"$\\Delta E_1 = {E0+delta1:.1f},\\ \\Delta E_2 = {E0+delta2:.1f},\\ \\Delta E_3 = {E0+delta3:.1f}$\n"
             f"$\\beta_2 = {beta2},\\ \\beta_3 = {beta3}$")

    # 创建 Figure 文件夹
    output_dir = os.path.join(save_path, "Figure")
    os.makedirs(output_dir, exist_ok=True)

    # 绘制 n 曲线
    fig_n, ax_n = plt.subplots(1, 1, figsize=(12, 9))
    plot_n_curves(ax_n, r, n1, n2, n3, title)
    plt.tight_layout()
    n_output = os.path.join(output_dir, "3_domain_results_n.png")
    plt.savefig(n_output, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig_n)
    print(f"n 曲线已保存至: {n_output}")

    # 绘制 force 曲线
    fig_f, ax_f = plt.subplots(1, 1, figsize=(12, 9))
    plot_force_curves(ax_f, r, force1, force2, force3, title)
    plt.tight_layout()
    f_output = os.path.join(output_dir, "3_domain_results_force.png")
    plt.savefig(f_output, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig_f)
    print(f"force 曲线已保存至: {f_output}")

if __name__ == "__main__":
    main()
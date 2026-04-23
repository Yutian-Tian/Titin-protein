import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
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
xi_f = 10.0  # 折叠态持续长度
k = 2.0      # 解折叠系数
E0 = 1.0     # 能量基准值
Ek = 2.0     # 能量系数

def energy_term_U(n_i, DeltaEi):
    """能量项: U(n_i) = ΔE_i n_i - ΔE_i cos(2π n_i)"""
    return DeltaEi * n_i - DeltaEi * np.cos(2 * np.pi * n_i)

def contour_length_Lci(n_i, xi_fi):
    """轮廓长度: L_{ci}(n_i) = ξ_fi + n_i (ξ_ui - ξ_fi)"""
    xi_ui = k * xi_fi
    return xi_fi + n_i * (xi_ui - xi_fi)

def end_to_end_factor_x_i(r_i, n_i, xi_fi):
    """端到端因子: x_i(r_i, n_i) = r_i / L_{ci}(n_i)"""
    L_ci = contour_length_Lci(n_i, xi_fi)
    return r_i / L_ci

def WLC_free_energy(x_i, L_ci):
    """WLC自由能: F_{WLC}(x_i, n_i) = (1/4) L_{ci} * [x_i^2 (3 - 2x_i) / (1 - x_i)]"""
    if x_i >= 0.999:
        return float('inf')
    return 0.25 * L_ci * (x_i**2 * (3.0 - 2.0 * x_i) / (1.0 - x_i))

def single_domain_free_energy(r_i, n_i, xi_fi, f_ext=0.0):
    """单个domain的自由能: F_d(x_i, n_i) = F_{WLC}(x_i, n_i) + U(n_i) - f_ext * x_i * L_{ci}"""
    DeltaEi = E0 + Ek * (xi_fi - 5.0)
    L_ci = contour_length_Lci(n_i, xi_fi)
    x_i = end_to_end_factor_x_i(r_i, n_i, xi_fi)
    F_wlc = WLC_free_energy(x_i, L_ci)
    Ui = energy_term_U(n_i, DeltaEi)
    work_term = f_ext * x_i * L_ci
    return F_wlc + Ui - work_term

def PlotMS(L_c, lineType ):
    r = np.linspace(0, 0.95*L_c, 1000)
    x = r/L_c
    x = np.asarray(x)
    L_c = np.asarray(L_c)
    
    # 初始化结果数组
    result = np.zeros_like(x, dtype=float)
    
    # 处理x < 1的情况
    mask = x < 1.0
    if np.any(mask):
        x_masked = x[mask]
        result[mask] = 0.25 * ((1 - x_masked)**(-2) - 1 + 4*x_masked)
    
    # 处理x >= 1的情况
    result[~mask] = np.inf

    line = plt.plot(r, result, lineType, color='purple', linewidth=lines_linewidth, label=f'$L_c = {L_c}$', zorder=3)

    return line

def Plot2state(r_fold, r_unfold, f, lineType):

    Et = single_domain_free_energy(r_unfold, 1.0, 10.0) - single_domain_free_energy(r_fold, 0.0, 10.0) 
    print(Et)
    p_u = 1 - 1/(1 + np.exp(-Et - f * (r_fold - r_unfold)))
    line = plt.plot(f, p_u, lineType, color='purple', linewidth=lines_linewidth, label=f'$\Delta E_t = {Et:.1f}$, $r_s = {r_fold - r_unfold:.1f}$', zorder=3)
    
    return line

def process_data(data_file):
    """
    处理包含所有数据的CSV文件
    
    参数:
    data_file: 数据文件路径，包含f, r_opt, n_opt, Fd_min, x_opt列
    
    返回:
    f: f值数组
    r_opt: r_opt数组
    n_opt: n_opt数组
    df: 原始数据DataFrame
    """
    try:
        print(f"正在读取文件: {data_file}")
        
        # 读取CSV文件，第一行作为列名
        df = pd.read_csv(data_file, header=0)
        
        print(f"数据形状: {df.shape}")
        print(f"数据列名: {df.columns.tolist()}")
        
        # 提取所需的列
        # 注意：根据您的描述，列顺序为f, r_opt, n_opt, Fd_min, x_opt
        f = df['f'].astype(float).values
        r_opt = df['r_opt'].astype(float).values
        n_opt = df['n_opt'].astype(float).values
        
        print(f"数据处理完成:")
        print(f"  f值数量: {len(f)}")
        print(f"  r_opt数量: {len(r_opt)}")
        print(f"  n_opt数量: {len(n_opt)}")
        
        # 显示一些统计信息
        print(f"\n统计信息:")
        print(f"  f范围: [{f.min():.4f}, {f.max():.4f}]")
        print(f"  r_opt范围: [{r_opt.min():.4f}, {r_opt.max():.4f}]")
        print(f"  n_opt范围: [{n_opt.min():.4f}, {n_opt.max():.4f}]")
        
        # 显示前几行数据以供验证
        print(f"\n前5行数据预览:")
        print(df.head())
        
        return f, r_opt, n_opt, df
        
    except FileNotFoundError as e:
        print(f"错误: 未找到文件 - {e}")
        return None, None, None, None
    except KeyError as e:
        print(f"错误: 数据文件中缺少必要的列 - {e}")
        print(f"请确保数据文件包含以下列: f, r_opt, n_opt, Fd_min, x_opt")
        return None, None, None, None
    except Exception as e:
        print(f"处理数据时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def create_f_r_scatter(f, r_opt, save_dir=None, save_name='f_r_scatter.png'):
    """
    创建f-r_opt散点图
    
    参数:
    f: f值数组
    r_opt: r_opt数组
    save_dir: 保存目录路径
    save_name: 保存文件名
    
    返回:
    fig: 图形对象
    """
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    PlotMS(10.0, "-")
    PlotMS(20.0, "--")
    # 使用散点图绘制f-r_opt关系：r_opt在x轴，f在y轴
    scatter = ax.scatter(r_opt, f, 
                        s=lines_markersize*3,  # 散点大小
                        c='red',               # 散点颜色
                        alpha=0.7,             # 透明度
                        edgecolors='red',      # 边缘颜色
                        linewidths=1,          # 边缘宽度
                        label='$r_{{opt}}$', 
                        zorder=3)


    # 设置标签和标题
    ax.set_xlabel('End-to-end distance $r$', fontsize=label_fontsize)
    ax.set_ylabel('Force $f$', fontsize=label_fontsize)
    ax.set_title('$f$ - $r$ Relationship', 
                fontsize=title_fontsize, pad=20)
    
    # 设置网格
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    
    # 设置图例
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, 
             edgecolor='none', loc='best')
    
    # 设置坐标轴范围（根据数据自动调整）
    x_min, x_max = r_opt.min(), r_opt.max()
    y_min, y_max = f.min(), f.max()
    
    # 添加一些边距
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
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
    
    # 保存图形
    if save_dir:
        save_path = os.path.join(save_dir, save_name)
        fig.savefig(save_path, dpi=savefig_dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"f-r散点图已保存至: {save_path}")
    
    return fig

def create_n_f_scatter(f, n_opt, save_dir=None, save_name='n_f_scatter.png'):
    """
    创建n_opt-f散点图
    
    参数:
    f: f值数组
    n_opt: n_opt数组
    save_dir: 保存目录路径
    save_name: 保存文件名
    
    返回:
    fig: 图形对象
    """
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    Plot2state(6.53, 13.06, f,"--")
    Plot2state(0.0, 13.06, f,"-")
    # 使用散点图绘制n_opt-f关系：f在x轴，n_opt在y轴
    scatter = ax.scatter(f, n_opt, 
                        s=lines_markersize*3,  # 散点大小
                        c='blue',              # 散点颜色
                        alpha=0.7,             # 透明度
                        edgecolors='blue',     # 边缘颜色
                        linewidths=1,          # 边缘宽度
                        label='$n_{{opt}}$', 
                        zorder=3)
    
    # 设置标签和标题
    ax.set_xlabel('Force $f$', fontsize=label_fontsize)
    ax.set_ylabel('Unfolded Number $n$', fontsize=label_fontsize)
    ax.set_title('$n$ - $f$ Relationship', 
                fontsize=title_fontsize, pad=20)
    
    # 设置网格
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    
    # 设置图例
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, 
             edgecolor='none', loc='best')
    
    # 设置坐标轴范围（根据数据自动调整）
    x_min, x_max = f.min(), f.max()
    y_min, y_max = n_opt.min(), n_opt.max()
    
    # 添加一些边距
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
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
    
    # 保存图形
    if save_dir:
        save_path = os.path.join(save_dir, save_name)
        fig.savefig(save_path, dpi=savefig_dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"n-f散点图已保存至: {save_path}")
    
    return fig

def main():
    """主程序"""
    print("=" * 80)
    print("单Domain数据处理和可视化程序")
    print("=" * 80)
    
    # ============ 在这里指定文件路径 ============
    # 修改这些路径为您实际的路径
    data_file_path = "/home/tyt/project/Single-chain/opt+R/Rand_xi/simulation_results/1_domain_IMS/single_domain_results_refined.csv"
    output_dir = "/home/tyt/project/Single-chain/opt+R/Rand_xi/simulation_results/1_domain_IMS"  # 保存结果的目录
    
    # 使用命令行参数（如果有）
    if len(sys.argv) > 1:
        data_file_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    print(f"数据文件路径: {data_file_path}")
    print(f"输出目录: {output_dir}")
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")
    
    # 处理数据
    f, r_opt, n_opt, df = process_data(data_file_path)
    
    if f is None:
        print("数据处理失败，程序退出")
        return
    
    # 创建可视化图表
    print("\n" + "=" * 80)
    print("创建散点图...")
    print("=" * 80)
    
    # 创建f-r散点图
    print("创建f-r散点图...")
    fig1 = create_f_r_scatter(f, r_opt, save_dir=output_dir, save_name='f_r_scatter.png')
    
    # 创建n-f散点图
    print("创建n-f散点图...")
    fig2 = create_n_f_scatter(f, n_opt, save_dir=output_dir, save_name='n_f_scatter.png')
    
    # 显示所有图形
    plt.show()
    
    print("\n" + "=" * 80)
    print("程序执行完毕!")
    print(f"所有散点图已保存至: {os.path.abspath(output_dir)}")
    print("=" * 80)

# ============ 运行主程序 ============
if __name__ == "__main__":
    main()
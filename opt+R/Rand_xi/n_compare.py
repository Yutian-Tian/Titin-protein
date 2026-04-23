"""
系统设置：N个折叠domain串联起来的链
结构异质性：折叠长度xi_f服从高斯分布
本程序用于可视化n-r
"""

"""
系统设置：N个折叠domain串联起来的链
结构异质性：解折叠能量惩罚DeltaE服从高斯分布
本程序用于可视化n-r关系
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.interpolate import interp1d

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

def load_and_process_data(r_file_path, n_file_path, fill_value=5.0):
    """
    加载并处理数据
    
    参数:
    r_file_path: r_values.csv文件路径
    n_file_path: n_values.csv文件路径
    fill_value: 插值时用于填充缺失数据的值，默认为5.0
    
    返回:
    r_common: 公共网格
    n_avg: 平均n值
    all_interpolated_n: 所有插值后的n数据
    num_groups: 组数
    """
    # Step 1: 读取数据
    print(f"正在读取数据...")
    print(f"r_values文件: {r_file_path}")
    print(f"n_values文件: {n_file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(r_file_path):
        print(f"错误: r_values.csv文件不存在于 {r_file_path}")
        return None, None, None, None
    
    if not os.path.exists(n_file_path):
        print(f"错误: n_values.csv文件不存在于 {n_file_path}")
        return None, None, None, None
    
    # 读取CSV文件
    try:
        r_data = pd.read_csv(r_file_path, header=None)
        n_data = pd.read_csv(n_file_path, header=None)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None, None, None, None
    
    print(f"r_values数据形状: {r_data.shape}")
    print(f"n_values数据形状: {n_data.shape}")
    
    # 检查数据形状是否匹配
    if r_data.shape[1] != n_data.shape[1]:
        print("警告: r和n数据的列数不匹配!")
        # 如果r只有一列，n有多列，则使用同一列r数据
        if r_data.shape[1] == 1 and n_data.shape[1] > 1:
            print("r数据只有一列，将使用同一列r数据对应所有n列")
            # 复制r列以匹配n的列数
            r_data = pd.concat([r_data] * n_data.shape[1], axis=1, ignore_index=True)
        else:
            # 取最小列数
            min_cols = min(r_data.shape[1], n_data.shape[1])
            r_data = r_data.iloc[:, :min_cols]
            n_data = n_data.iloc[:, :min_cols]
            print(f"使用前{min_cols}列数据")
    
    num_groups = r_data.shape[1]
    print(f"数据组数: {num_groups}")
    
    # Step 2: 建立公共网格
    print("\n计算最大r值...")
    r_max_values = []
    for i in range(num_groups):
        # 移除NaN值
        r_col = r_data.iloc[:, i].dropna().values
        if len(r_col) > 0:
            r_max = np.max(r_col)
            r_max_values.append(r_max)
        else:
            print(f"  第{i+1}组: 无有效数据")
    
    if not r_max_values:
        print("错误: 没有有效的数据!")
        return None, None, None, None
    
    max_r_max = np.max(r_max_values)
    print(f"\n所有组中最大的r_max: {max_r_max:.4f}")
    
    # 建立公共网格
    r_common = np.linspace(0, max_r_max, 1000)
    print(f"公共网格: 0 到 {max_r_max:.4f}, 共{len(r_common)}个点")
    
    # Step 3: 分段线性插值
    print("\n进行分段线性插值...")
    all_interpolated_n = []
    
    for i in range(num_groups):
        # 获取当前组的r和n数据，移除NaN值
        r_col = r_data.iloc[:, i].dropna().values
        n_col = n_data.iloc[:, i].dropna().values
        
        # 确保r和n长度相同
        min_len = min(len(r_col), len(n_col))
        if min_len == 0:
            print(f"  第{i+1}组: 跳过，无有效数据")
            continue
        
        r_col = r_col[:min_len]
        n_col = n_col[:min_len]
        
        # 确保r是递增的（对数据进行排序）
        sort_idx = np.argsort(r_col)
        r_sorted = r_col[sort_idx]
        n_sorted = n_col[sort_idx]
        
        # 创建插值函数
        try:
            interp_func = interp1d(r_sorted, n_sorted, kind='linear', 
                                 bounds_error=False, fill_value=fill_value)
            
            # 在公共网格上插值
            n_interp = interp_func(r_common)
            all_interpolated_n.append(n_interp)
        except Exception as e:
            print(f"  第{i+1}组: 插值失败 - {e}")
    
    if not all_interpolated_n:
        print("错误: 没有成功插值的数据!")
        return None, None, None, None
    
    # 将所有插值数据组合成数组
    all_interpolated_n = np.array(all_interpolated_n)
    print(f"\n成功插值{len(all_interpolated_n)}组数据")
    
    # 计算平均值
    print("计算平均值...")
    n_avg = np.mean(all_interpolated_n, axis=0)
    
    # 计算标准差（可选）
    n_std = np.std(all_interpolated_n, axis=0)
    
    print("数据处理完成!")
    
    return r_common, n_avg, all_interpolated_n, num_groups

def visualize_data(r_common, n_avg, all_interpolated_n, num_groups, save_path=None):
    """
    可视化数据
    
    参数:
    r_common: 公共网格
    n_avg: 平均n值
    all_interpolated_n: 所有插值后的n数据
    num_groups: 组数
    save_path: 保存路径，如果为None则不保存
    """
    print("\n开始可视化...")
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    # 绘制原始轨迹（半透明灰细线）
    for i in range(len(all_interpolated_n)):
        ax.plot(r_common, all_interpolated_n[i], 
                color='gray', alpha=0.3, linewidth=0.8, zorder=1)
    
    # 绘制平均曲线（红色粗线）
    ax.plot(r_common, n_avg, 
            color='red', linewidth=lines_linewidth, 
            label=f'Average of {num_groups} curves', zorder=3)
    
    # 设置标签和标题
    ax.set_xlabel('End-to-end distance $r$', fontsize=label_fontsize)
    ax.set_ylabel('Unfolding number $n_u$', fontsize=label_fontsize)
    ax.set_title(f'Unfolding number $n_u$ vs. distance $r$', 
                fontsize=title_fontsize, pad=20)
    
    # 设置网格
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    
    # 设置图例
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, 
              edgecolor='none', loc='best')
    
    # 设置坐标轴范围
    ax.set_xlim(0, np.max(r_common) * 0.95)
    
    # 自动调整y轴范围
    y_min = np.min(all_interpolated_n)
    y_max = np.max(all_interpolated_n)
    y_padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
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
    if save_path:
        # 确保目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(save_path, dpi=savefig_dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"图形已保存至: {save_path}")
    
    return fig

def main():
    """主程序"""
    print("=" * 80)
    print("n-r关系数据分析和可视化程序")
    print("=" * 80)
    
    # ============ 在这里指定文件路径 ============
    # 请根据您的实际情况修改这些路径
    r_file_path = "/home/tyt/project/Single-chain/opt+R/Rand_xi/simulation_results/r_values.csv"  # 替换为您的r_values.csv文件路径
    n_file_path = "/home/tyt/project/Single-chain/opt+R/Rand_xi/simulation_results/n_values.csv"  # 替换为您的n_values.csv文件路径
    output_path = "/home/tyt/project/Single-chain/opt+R/Rand_xi/simulation_results/n_r_analysis.png"  # 替换为您想要保存的输出路径
    fill_value = 10.0  # 插值时用于填充缺失数据的值
    
    print(f"指定的r_values.csv路径: {r_file_path}")
    print(f"指定的n_values.csv路径: {n_file_path}")
    print(f"输出文件路径: {output_path}")
    print(f"填充值: {fill_value}")
    
    # Step 2-3: 加载和处理数据
    r_common, n_avg, all_interpolated_n, num_groups = load_and_process_data(
        r_file_path, n_file_path, fill_value=fill_value
    )
    
    if r_common is None:
        print("数据处理失败，程序退出")
        return
    
    # Step 4: 可视化
    fig = visualize_data(r_common, n_avg, all_interpolated_n, num_groups, save_path=output_path)
    
    # 输出统计信息
    print("\n" + "=" * 80)
    print("统计信息:")
    print(f"  数据组数: {num_groups}")
    print(f"  公共网格点数: {len(r_common)}")
    print(f"  平均曲线有效点数: {len(n_avg)}")
    print(f"  平均n值范围: {np.min(n_avg):.4f} 到 {np.max(n_avg):.4f}")
    print(f"  平均n值在最大r处的值: {n_avg[-1]:.4f}")
    print("=" * 80)

# ============ 运行主程序 ============
if __name__ == "__main__":
    main()
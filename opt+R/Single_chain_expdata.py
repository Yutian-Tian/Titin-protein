"""
根据实验数据绘制曲线，实验数据来源于文献
可视化风格：Times New Roman字体、大字号、内部刻度、粗线宽、网格线等
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

# ============ 字体路径（如有需要可修改） ============
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

# ============ 样式变量定义（参考提供的代码） ============
font_family = 'Times New Roman'
font_weight = 'normal'
math_fontset = 'stix'
math_rm = 'Times New Roman'
math_it = 'Times New Roman:italic'
math_bf = 'Times New Roman:bold'

title_fontsize = 35          # 此处统一使用参考代码中的字号，若需调整可修改
label_fontsize = 35
tick_fontsize = 35
legend_fontsize = 20
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

marker_size = 5

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
else:
    plt.rcParams['font.family'] = font_family

plt.rcParams.update({
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
    'ytick.major.size': xtick_major_size,
    'ytick.major.width': ytick_major_width,
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

def plot_excel_rf(file_path, output_image=None):
    """
    读取Excel文件中前4个Sheet的r和力f数据，并绘制在同一张图上（采用学术论文风格）

    参数:
        file_path (str): Excel文件路径
        output_image (str, optional): 若提供，则保存图片到该路径；否则显示图片
    """
    # 读取所有Sheet名称
    try:
        excel_data = pd.ExcelFile(file_path)
        sheet_names = excel_data.sheet_names
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到，请检查路径")
        return
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return

    # 限制最多使用4个Sheet
    sheets_to_use = sheet_names[:4]
    if len(sheet_names) < 4:
        print(f"提示：文件中仅有 {len(sheet_names)} 个Sheet，将全部使用")

    # 创建画布（参考代码中图形大小为12x9英寸）
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    # 自动分配颜色（使用tab10色板）
    colors = plt.cm.tab10(np.linspace(0, 1, len(sheets_to_use)))

    # 遍历每个Sheet并绘图
    for idx, sheet_name in enumerate(sheets_to_use):
        try:
            # 读取Sheet，第一行作为列名
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # 检查是否有至少两列数据
            if df.shape[1] < 2:
                print(f"警告：Sheet '{sheet_name}' 列数不足2，跳过")
                continue
            
            # 提取前两列（第一列为r，第二列为力f）
            r_raw = df.iloc[:, 0]
            f_raw = df.iloc[:, 1]
            
            # 转换为数值类型，无效值变为NaN
            r = pd.to_numeric(r_raw, errors='coerce')
            f = pd.to_numeric(f_raw, errors='coerce')
            
            # 删除包含NaN的行
            valid_mask = r.notna() & f.notna()
            r_clean = r[valid_mask]
            f_clean = f[valid_mask]
            
            if len(r_clean) == 0:
                print(f"警告：Sheet '{sheet_name}' 无有效数值数据，跳过")
                continue
            
        
            # 绘制曲线（带圆形标记，线宽和标记大小由全局rcParams控制）
            ax.plot(r_clean, f_clean, 
                    marker='o', 
                    linestyle='-', 
                    color=colors[idx],
                    label=sheet_name,
                    markersize = marker_size)
            
            print(f"成功加载 Sheet '{sheet_name}': {len(r_clean)} 个有效数据点")
            
        except Exception as e:
            print(f"处理 Sheet '{sheet_name}' 时出错：{e}")
            continue

    # 图表装饰（样式与参考代码一致）
    ax.set_xlabel('Distance $r$', fontsize=label_fontsize)
    ax.set_ylabel('force $f$', fontsize=label_fontsize)
    ax.set_title('Single chain $f-r$ Curves', fontsize=title_fontsize, pad=20)
    
    # 图例（位置最佳，无边框）
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    
    # 网格（虚线，透明度由grid_alpha控制）
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    
    # 刻度设置：内部、顶部/右侧显示、主次刻度
    ax.tick_params(axis='both', which='major',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width, length=xtick_major_size)
    ax.tick_params(axis='both', which='minor',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width*0.75, length=xtick_major_size*0.5)
    ax.minorticks_on()
    
    # 设置坐标轴边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)
    
    ax.set_xlim(-50,600)
    ax.set_ylim(-600,2500)
    
    # 输出结果
    if output_image:
        plt.savefig(output_image, dpi=savefig_dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"图表已保存至：{output_image}")
    else:
        plt.show()
    
    # 关闭图形，释放内存
    plt.close(fig)

# 使用示例
if __name__ == "__main__":
    # 请将以下路径替换为您的Excel文件实际路径
    excel_file_path = "/home/tyt/project/Single-chain/opt+R/Protein unfolding data.xlsx"
    output_path = "/home/tyt/project/Single-chain/opt+R/exp_data_figure.png"
    plot_excel_rf(excel_file_path, output_image=output_path)
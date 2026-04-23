# 曲线轨迹绘制为基于网格统计的灰度背景

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


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


def create_grid_heatmap_background(all_curves_data, grid_size=(100, 100), cmap='gray_r'):
    """
    创建基于网格统计的灰度背景
    
    参数:
    - all_curves_data: 所有曲线的数据列表，每个元素为(r, f)元组
    - grid_size: 网格划分大小 (x_grid, y_grid)
    - cmap: 使用的颜色映射（灰度）
    """
    if not all_curves_data:
        return
    
    # 确定整个数据范围
    all_r_vals = []
    all_f_vals = []
    for r_vals, f_vals in all_curves_data:
        all_r_vals.extend(r_vals)
        all_f_vals.extend(f_vals)
    
    if not all_r_vals or not all_f_vals:
        return
    
    r_min, r_max = min(all_r_vals), max(all_r_vals)
    f_min, f_max = min(all_f_vals), max(all_f_vals)
    
    # 扩展边界以防止边缘效应
    r_range = r_max - r_min
    f_range = f_max - f_min
    r_min -= r_range * 0.05
    r_max += r_range * 0.05
    f_min -= f_range * 0.05
    f_max += f_range * 0.05
    
    # 创建网格
    x_grid, y_grid = grid_size
    x_edges = np.linspace(r_min, r_max, x_grid + 1)
    y_edges = np.linspace(f_min, f_max, y_grid + 1)
    
    # 初始化计数网格
    count_grid = np.zeros((y_grid, x_grid))
    
    # 统计每个网格中的点数
    for r_vals, f_vals in all_curves_data:
        # 将每个点分配到对应的网格
        for r_val, f_val in zip(r_vals, f_vals):
            # 找到r值所在的列索引
            r_idx = np.searchsorted(x_edges, r_val) - 1
            # 找到f值所在的行索引
            f_idx = np.searchsorted(y_edges, f_val) - 1
            
            # 确保索引在有效范围内
            if 0 <= r_idx < x_grid and 0 <= f_idx < y_grid:
                count_grid[f_idx, r_idx] += 1
    
    # 将对数化计数用于更好的可视化
    # 避免log(0)错误
    count_grid_log = np.log10(count_grid + 1)
    
    # 归一化到0-1范围用于灰度显示
    if count_grid_log.max() > count_grid_log.min():
        normalized_grid = (count_grid_log - count_grid_log.min()) / (count_grid_log.max() - count_grid_log.min())
    else:
        normalized_grid = np.zeros_like(count_grid_log)
    
    # 创建网格的中心点坐标
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
    # 绘制灰度背景
    plt.pcolormesh(x_edges, y_edges, normalized_grid, 
                  cmap=cmap, alpha=0.7, shading='auto')
    
    return normalized_grid


def create_high_density_heatmap(all_curves_data, grid_size=(200, 200), sigma=1.0):
    """
    创建高密度热图背景（使用高斯模糊平滑）
    
    参数:
    - all_curves_data: 所有曲线的数据列表
    - grid_size: 网格划分大小
    - sigma: 高斯平滑的标准差
    """
    if not all_curves_data:
        return
    
    # 确定整个数据范围
    all_r_vals = []
    all_f_vals = []
    for r_vals, f_vals in all_curves_data:
        all_r_vals.extend(r_vals)
        all_f_vals.extend(f_vals)
    
    if not all_r_vals or not all_f_vals:
        return
    
    r_min, r_max = min(all_r_vals), max(all_r_vals)
    f_min, f_max = min(all_f_vals), max(all_f_vals)
    
    # 扩展边界
    r_range = r_max - r_min
    f_range = f_max - f_min
    r_min -= r_range * 0.02
    r_max += r_range * 0.02
    f_min -= f_range * 0.02
    f_max += f_range * 0.02
    
    # 创建高分辨率网格
    x_grid, y_grid = grid_size
    density = np.zeros((y_grid, x_grid))
    
    # 转换函数：将坐标映射到网格索引
    def to_grid_idx(val, min_val, max_val, grid_dim):
        return int((val - min_val) / (max_val - min_val) * (grid_dim - 1))
    
    # 统计密度
    for r_vals, f_vals in all_curves_data:
        for r_val, f_val in zip(r_vals, f_vals):
            i = to_grid_idx(r_val, r_min, r_max, x_grid)
            j = to_grid_idx(f_val, f_min, f_max, y_grid)
            if 0 <= i < x_grid and 0 <= j < y_grid:
                density[j, i] += 1
    
    # 应用高斯模糊（可选）
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        density = gaussian_filter(density, sigma=sigma)
    
    # 对数变换增强可视化
    density_log = np.log10(density + 1)
    
    # 归一化
    if density_log.max() > density_log.min():
        density_normalized = (density_log - density_log.min()) / (density_log.max() - density_log.min())
    else:
        density_normalized = density_log
    
    # 创建坐标网格
    x = np.linspace(r_min, r_max, x_grid)
    y = np.linspace(f_min, f_max, y_grid)
    X, Y = np.meshgrid(x, y)
    
    # 绘制热图
    plt.pcolormesh(X, Y, density_normalized, cmap='gray_r', alpha=0.6, shading='auto')
    
    return density_normalized


def plot_f_r_curves(df1, df2, f_upper_limit=None, title="f-r Curves", save_figure=False):
    """
    绘制f-r曲线
    修改：使用网格统计创建灰度背景
    """
    if df1 is None or df2 is None:
        print("DataFrames are empty. Cannot plot.")
        return
    
    # 获取列数，确保两个文件列数相同
    num_columns = min(len(df1.columns), len(df2.columns))
    
    # 用于平均值计算的数据（保留所有数据）
    all_curves_data_full = []
    # 用于可视化的数据（过滤后）
    all_curves_data_filtered = []
    
    # 统计信息
    filtered_points = 0
    kept_points = 0
    
    # 首先处理所有数据，准备用于平均值计算
    for i in range(num_columns):
        r = df1.iloc[:, i].dropna().values
        f = df2.iloc[:, i].dropna().values
        
        # 确保r和f长度相同
        min_len = min(len(r), len(f))
        r = r[:min_len]
        f = f[:min_len]
        
        # 保存所有数据用于平均值计算
        all_curves_data_full.append((r.copy(), f.copy()))
        
        # 为可视化过滤数据
        if f_upper_limit is not None:
            mask = f < f_upper_limit
            filtered_points += sum(~mask)
            kept_points += sum(mask)
            
            r_filtered = r[mask]
            f_filtered = f[mask]
        else:
            r_filtered = r.copy()
            f_filtered = f.copy()
            kept_points += len(r)
        
        # 如果过滤后仍有数据，则保存用于可视化
        if len(r_filtered) > 2:  # 至少需要2个点才能进行线性插值
            all_curves_data_filtered.append((r_filtered, f_filtered))
    
    if len(all_curves_data_filtered) == 0:
        print("No valid data after filtering.")
        return
    
    # ========== 平均值计算部分（使用所有数据） ==========
    # 确定用于平均值计算的r值范围
    all_r_min_full = []
    all_r_max_full = []
    for r_orig, f_orig in all_curves_data_full:
        all_r_min_full.append(np.min(r_orig))
        all_r_max_full.append(np.max(r_orig))
    
    # 确定用于平均值计算的统一r值范围
    r_min_full = min(all_r_min_full)
    r_max_full = max(all_r_max_full)
    
    # 生成用于平均值计算的统一r值数组
    r_uniform_full = np.linspace(r_min_full, r_max_full, 10000)
    fsum_full = np.zeros_like(r_uniform_full)
    count_full = np.zeros_like(r_uniform_full)
    
    # 对所有数据进行插值并求和
    for r_orig, f_orig in all_curves_data_full:
        # 线性插值
        interp_func = interpolate.interp1d(r_orig, f_orig, kind='linear', 
                                          bounds_error=False, fill_value=np.nan)
        
        # 在统一r值下计算插值f值
        f_interp = interp_func(r_uniform_full)
        
        # 累加有效值
        valid_mask = ~np.isnan(f_interp)
        fsum_full[valid_mask] += f_interp[valid_mask]
        count_full[valid_mask] += 1
    
    # 计算平均值（使用所有数据）
    with np.errstate(divide='ignore', invalid='ignore'):
        f_avg_full = np.where(count_full > 0, fsum_full / count_full, np.nan)
    
    # ========== 确定可视化范围 ==========
    # 确定可视化的r值范围（使用过滤后数据）
    all_r_min = []
    all_r_max = []
    for r_vals, _ in all_curves_data_filtered:
        all_r_min.append(np.min(r_vals))
        all_r_max.append(np.max(r_vals))
    
    if all_r_min and all_r_max:
        common_r_min = 0  # 取所有过滤后曲线r值的最小值中的最大值
        common_r_max = 450  # 取所有过滤后曲线r值的最大值中的最小值
    else:
        # 如果没有过滤后的数据，使用平均值曲线的范围
        common_r_min = r_min_full
        common_r_max = r_max_full
    
    # ========== 图1：过滤后的曲线（带限制） ==========
    plt.figure(figsize=(12, 8))
    
    # ========== 修改部分：创建基于网格统计的灰度背景 ==========
    print("Creating grid-based grayscale background...")
    
    # 方法1：简单网格统计（推荐用于大多数情况）
    #heatmap_data = create_grid_heatmap_background(
    #    all_curves_data_filtered, 
    #    grid_size=(500, 500),  # 150x150的网格
    #    cmap='gray_r'  # 反转的灰度色图（点数越多颜色越深）
    #)
    
    # 方法2：高密度热图（可选，更平滑）
    heatmap_data = create_high_density_heatmap(
        all_curves_data_filtered,
        grid_size=(200, 200),
        sigma=3.0
    )
    
    # ========== 绘制平均值曲线（只显示到上限值以下） ==========
    # 找到平均值曲线在可视化范围内的部分
    viz_mask_full = (r_uniform_full >= common_r_min) & (r_uniform_full <= common_r_max)
    r_avg_viz = r_uniform_full[viz_mask_full]
    f_avg_viz = f_avg_full[viz_mask_full]
    
    # 设置平均值曲线的样式
    AVG_COLOR = 'red'       # 平均值用红色突出显示
    AVG_LINEWIDTH = 2.5     # 平均值线宽加粗
    
    # 如果有限制，进一步过滤平均值曲线
    if f_upper_limit is not None:
        # 创建平均值曲线的插值函数
        avg_interp_func = interpolate.interp1d(r_avg_viz, f_avg_viz, kind='linear',
                                             bounds_error=False, fill_value=np.nan)
        
        # 在可视化范围内均匀采样
        r_samples = np.linspace(common_r_min, common_r_max, 5000)
        f_samples = avg_interp_func(r_samples)
        
        # 找出f值在限制以下的连续区间
        below_limit_mask = f_samples < f_upper_limit
        
        # 找出连续段
        segments = []
        start_idx = None
        
        for idx, is_below in enumerate(below_limit_mask):
            if is_below and start_idx is None:
                start_idx = idx
            elif not is_below and start_idx is not None:
                segments.append((start_idx, idx-1))
                start_idx = None
        
        # 处理最后一个段
        if start_idx is not None:
            segments.append((start_idx, len(below_limit_mask)-1))
        
        # 绘制每个连续段
        for start_idx, end_idx in segments:
            if end_idx - start_idx > 0:  # 至少需要2个点
                segment_r = r_samples[start_idx:end_idx+1]
                segment_f = f_samples[start_idx:end_idx+1]
                plt.plot(segment_r, segment_f, color=AVG_COLOR, 
                        linewidth=AVG_LINEWIDTH, label='Average' if start_idx == segments[0][0] else "")
        
        # 添加图例（只在第一个段添加了标签）
        if segments:
            plt.legend(fontsize=10, loc='best')
    else:
        # 如果没有限制，直接绘制整个平均值曲线
        plt.plot(r_avg_viz, f_avg_viz, color=AVG_COLOR, 
                linewidth=AVG_LINEWIDTH, label=f'Average (n={len(all_curves_data_full)})')
        plt.legend(fontsize=10, loc='best')
    
    # ========== 设置图形属性 ==========
    plt.xlabel('r', fontsize=14)
    plt.ylabel('f', fontsize=14)
    
    # 添加标题
    if f_upper_limit is not None and filtered_points > 0:
        plt.title(f'{title} (f < {f_upper_limit})', fontsize=16)
        print(f"Visualization: Filtered out {filtered_points} points with f >= {f_upper_limit}")
        print(f"Visualization: Kept {kept_points} points")
        print(f"Average calculation: Used all {kept_points + filtered_points} points")
    else:
        plt.title(title, fontsize=16)
        print(f"Used all {kept_points} points for both visualization and average")
    
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 设置y轴范围（如果需要）
    if f_upper_limit is not None:
        plt.ylim(0, f_upper_limit * 1.1)
    
    # 添加颜色条说明
    if heatmap_data is not None:
        plt.text(0.02, 0.98, 'Darker areas = Higher point density', 
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图形
    if save_figure:
        if f_upper_limit is not None:
            filename1 = f'/home/tyt/project/Single-chain/opt+R/M-S_Rand_L+E/column_format/f_r_curves_filtered{f_upper_limit}_grid_background.png'
        else:
            filename1 = '/home/tyt/project/Single-chain/opt+R/M-S_Rand_L+E/column_format/f_r_curves_grid_background.png'
        
        plt.savefig(filename1, dpi=300, bbox_inches='tight')
        print(f"\nFigure 1 saved as {filename1}")
        print(f"Grid background created from {len(all_curves_data_filtered)} curves")
    
    # ========== 图2：所有原始曲线（不过滤）的网格背景 ==========
    plt.figure(figsize=(12, 8))
    
    print("Creating grid-based grayscale background for all data...")
    
    # 为所有数据创建网格背景
    heatmap_all_data = create_grid_heatmap_background(
        all_curves_data_full, 
        grid_size=(1000, 1000),
        cmap='gray_r'
    )
    
    # 绘制完整的平均值曲线
    valid_avg_mask = ~np.isnan(f_avg_full)
    if np.any(valid_avg_mask):
        plt.plot(r_uniform_full[valid_avg_mask], f_avg_full[valid_avg_mask], 
                color=AVG_COLOR, linewidth=AVG_LINEWIDTH,
                label=f'Average (n={len(all_curves_data_full)})')
    
    # 设置图形属性
    plt.xlabel('r', fontsize=14)
    plt.ylabel('f', fontsize=14)
    plt.title(f'{title} (All Data, No Filtering)', fontsize=16)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加颜色条说明
    if heatmap_all_data is not None:
        plt.text(0.02, 0.98, 'Darker areas = Higher point density', 
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存第二个图形
    if save_figure:
        filename2 = '/home/tyt/project/Single-chain/opt+R/M-S_Rand_L+E/column_format/f_r_curves_all_data_grid_background.png'
        plt.savefig(filename2, dpi=300, bbox_inches='tight')
        print(f"\nFigure 2 saved as {filename2}")
        print(f"Grid background created from {len(all_curves_data_full)} curves (all data)")
        print(f"Total data points: {kept_points + filtered_points}")

def main():
    # 设置文件路径（请根据实际情况修改）
    file1_path = '/home/tyt/project/Single-chain/opt+R/M-S_Rand_L+E/column_format/r_values.csv'
    file2_path = '/home/tyt/project/Single-chain/opt+R/M-S_Rand_L+E/column_format/f_values.csv'
    
    # 设置f的上限值（仅用于可视化）
    f_upper_limit = 10.0  # 可视化时只显示f < 10的数据
    
    # 读取CSV文件
    df1, df2 = read_csv_files(file1_path, file2_path)
    
    # 绘制曲线
    plot_f_r_curves(df1, df2, 
                   f_upper_limit=f_upper_limit,
                   title="f vs r Curves", 
                   save_figure=True)
    
    print('\nProcess completed!')
    plt.show()

if __name__ == "__main__":
    main()
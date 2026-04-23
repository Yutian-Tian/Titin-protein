import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


def read_csv_files(r_file_path, f_file_path):
    """
    读取CSV文件
    r_file_path: 包含r值的文件，只有一列
    f_file_path: 包含f值的文件，每列代表一条轨迹
    """
    try:
        # 读取r值文件（只有一列）
        df_r = pd.read_csv(r_file_path)
        
        # 读取f值文件（多列）
        df_f = pd.read_csv(f_file_path)
        
        return df_r, df_f
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None
    except Exception as e:
        print(f"Error reading files: {e}")
        return None, None

def plot_f_r_curves(df_r, df_f, f_upper_limit=None, title="f-r Curves", save_figure=False):
    """
    绘制f-r曲线
    f_upper_limit: f值的上限，可视化时只显示f < f_upper_limit的数据
    平均值计算使用所有数据，但可视化时平均值也只显示到上限值以下
    """
    if df_r is None or df_f is None:
        print("DataFrames are empty. Cannot plot.")
        return
    
    # 获取r值（只有一列）
    r_values = df_r.iloc[:, 0].dropna().values
    num_r_points = len(r_values)
    
    # 获取f值的列数（轨迹数量）
    num_trajectories = len(df_f.columns)
    
    print(f"Number of r points: {num_r_points}")
    print(f"Number of trajectories: {num_trajectories}")
    
    # 用于平均值计算的数据（保留所有数据）
    all_curves_data_full = []
    # 用于可视化的数据（过滤后）
    all_curves_data_filtered = []
    
    # 统计信息
    filtered_points_total = 0
    kept_points_total = 0
    
    # 首先处理所有数据，准备用于平均值计算
    for i in range(num_trajectories):
        f = df_f.iloc[:, i].dropna().values
        
        # 确保r和f长度相同（取较小值）
        min_len = min(num_r_points, len(f))
        r_curr = r_values[:min_len]
        f_curr = f[:min_len]
        
        # 保存所有数据用于平均值计算
        all_curves_data_full.append((r_curr.copy(), f_curr.copy()))
        
        # 为可视化过滤数据
        if f_upper_limit is not None:
            mask = f_curr < f_upper_limit
            filtered_points = sum(~mask)
            kept_points = sum(mask)
            
            filtered_points_total += filtered_points
            kept_points_total += kept_points
            
            r_filtered = r_curr[mask]
            f_filtered = f_curr[mask]
        else:
            r_filtered = r_curr.copy()
            f_filtered = f_curr.copy()
            kept_points_total += len(r_curr)
        
        # 如果过滤后仍有数据，则保存用于可视化
        if len(r_filtered) > 1:  # 至少需要2个点才能进行线性插值
            all_curves_data_filtered.append((r_filtered, f_filtered))
    
    if len(all_curves_data_full) == 0:
        print("No data found.")
        return
    
    # ========== 平均值计算部分（使用所有数据） ==========
    # 确定用于平均值计算的r值范围
    # 所有曲线共享相同的r值，所以直接使用完整的r值范围
    r_min_full = np.min(r_values)
    r_max_full = np.max(r_values)
    
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
    # 对于过滤后的数据，确定共同的r值范围
    common_r_min = r_min_full
    common_r_max = r_max_full
    
    if all_curves_data_filtered:
        # 找到所有过滤后曲线的最小和最大r值
        all_r_min_filtered = [np.min(r) for r, _ in all_curves_data_filtered]
        all_r_max_filtered = [np.max(r) for r, _ in all_curves_data_filtered]
        
        common_r_min = max(all_r_min_filtered) if all_r_min_filtered else r_min_full
        common_r_max = min(all_r_max_filtered) if all_r_max_filtered else r_max_full
    
    # ========== 图1：过滤后的曲线（带限制） ==========
    # 设置图形
    plt.figure(figsize=(10, 6))
    
    # 生成用于可视化的统一r值数组
    r_uniform_viz = np.linspace(common_r_min, common_r_max, 10000)
    
    # ========== 原始数据的线绘制为半透明的灰线 ==========
    # 设置灰线的参数
    GRAY_COLOR = 'gray'  # 灰色
    GRAY_ALPHA = 0.1     # 透明度10%
    GRAY_LINEWIDTH = 1.0 # 线宽
    
    # 为每条过滤后的曲线绘制插值曲线（使用半透明灰线）
    for i, (r_orig, f_orig) in enumerate(all_curves_data_filtered):
        # 线性插值
        interp_func = interpolate.interp1d(r_orig, f_orig, kind='linear', 
                                          bounds_error=False, fill_value=np.nan)
        
        # 在统一r值下计算插值f值
        f_interp = interp_func(r_uniform_viz)
        
        # 绘制插值曲线（半透明灰线）
        valid_mask = ~np.isnan(f_interp)
        if np.any(valid_mask):
            plt.plot(r_uniform_viz[valid_mask], f_interp[valid_mask], 
                    color=GRAY_COLOR, linewidth=GRAY_LINEWIDTH, alpha=GRAY_ALPHA)
    
    # ========== 绘制平均值曲线（只显示到上限值以下） ==========
    # 找到平均值曲线在可视化范围内的部分
    viz_mask_full = (r_uniform_full >= common_r_min) & (r_uniform_full <= common_r_max)
    r_avg_viz = r_uniform_full[viz_mask_full]
    f_avg_viz = f_avg_full[viz_mask_full]
    
    # 设置平均值曲线的样式
    AVG_COLOR = 'blue'     # 平均值用蓝色
    AVG_LINEWIDTH = 2.0    # 平均值线宽
    
    # 如果有限制，进一步过滤平均值曲线，只显示f值在限制以下的部分
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
                plt.plot(segment_r, segment_f, color=AVG_COLOR, linewidth=AVG_LINEWIDTH)
        
        # 只在第一个段添加图例
        if segments:
            plt.plot([], [], color=AVG_COLOR, linewidth=AVG_LINEWIDTH, 
                    label=f'Average (n={len(all_curves_data_full)})')
    else:
        # 如果没有限制，直接绘制整个平均值曲线
        valid_mask = ~np.isnan(f_avg_viz)
        if np.any(valid_mask):
            plt.plot(r_avg_viz[valid_mask], f_avg_viz[valid_mask], 
                    color=AVG_COLOR, linewidth=AVG_LINEWIDTH, 
                    label=f'Average (n={len(all_curves_data_full)})')
    
    # ========== 设置图形属性 ==========
    plt.xlabel('r', fontsize=12)
    plt.ylabel('f', fontsize=12)
    
    # 添加图例
    plt.legend(fontsize=10, loc='best')
    
    # 添加过滤信息
    if f_upper_limit is not None and filtered_points_total > 0:
        plt.title(f'{title} (f < {f_upper_limit})', fontsize=14)
        print(f"\nVisualization: Filtered out {filtered_points_total} points with f >= {f_upper_limit}")
        print(f"Visualization: Kept {kept_points_total} points")
        print(f"Average calculation: Used all {kept_points_total + filtered_points_total} points")
        print(f"Average curve also truncated where f >= {f_upper_limit}")
    else:
        plt.title(title, fontsize=14)
        print(f"\nUsed all {kept_points_total} points for both visualization and average")
    
    plt.grid(True, alpha=0.3)
    
    # 设置y轴范围（如果需要）
    if f_upper_limit is not None:
        plt.ylim(0, f_upper_limit * 1.1)  # 留10%的边距
    
    # 保存图形（如果需要）
    if save_figure:
        if f_upper_limit is not None:
            filename1 = f'/home/tyt/project/Single-chain/opt+R/Multichain_results/f_r_curves_filtered{f_upper_limit}_gray.png'
        else:
            filename1 = '/home/tyt/project/Single-chain/opt+R/Multichain_results/f_r_curves_gray.png'
        
        plt.savefig(filename1, dpi=300, bbox_inches='tight')
        print(f"\nFigure 1 saved as {filename1}")
        print(f"Visualization r range: [{common_r_min:.4f}, {common_r_max:.4f}]")
        print(f"Average calculation r range: [{r_min_full:.4f}, {r_max_full:.4f}]")
        print(f"Number of individual curves: {len(all_curves_data_filtered)}")
    
    # ========== 图2：所有原始曲线（不过滤） ==========
    plt.figure(figsize=(10, 6))
    
    # 设置灰线的参数
    GRAY_COLOR = 'gray'  # 灰色
    GRAY_ALPHA = 0.1     # 透明度10%
    GRAY_LINEWIDTH = 1.0 # 线宽
    
    # 为每条原始曲线绘制插值曲线（半透明灰线）
    for i, (r_orig, f_orig) in enumerate(all_curves_data_full):
        if len(r_orig) > 1:  # 至少需要2个点
            # 线性插值
            interp_func = interpolate.interp1d(r_orig, f_orig, kind='linear', 
                                              bounds_error=False, fill_value=np.nan)
            
            # 生成该曲线的r值网格
            r_curve = np.linspace(np.min(r_orig), np.max(r_orig), 1000)
            f_interp = interp_func(r_curve)
            
            # 绘制插值曲线（半透明灰线）
            valid_mask = ~np.isnan(f_interp)
            if np.any(valid_mask):
                plt.plot(r_curve[valid_mask], f_interp[valid_mask], 
                        color=GRAY_COLOR, linewidth=GRAY_LINEWIDTH, alpha=GRAY_ALPHA)
    
    # ========== 绘制完整平均值曲线 ==========
    # 绘制完整的平均值曲线
    valid_avg_mask = ~np.isnan(f_avg_full)
    if np.any(valid_avg_mask):
        plt.plot(r_uniform_full[valid_avg_mask], f_avg_full[valid_avg_mask], 
                color=AVG_COLOR, linewidth=AVG_LINEWIDTH, 
                label=f'Average (n={len(all_curves_data_full)})')
    
    # ========== 设置图形属性 ==========
    plt.xlabel('r', fontsize=12)
    plt.ylabel('f', fontsize=12)
    plt.title(f'{title} (All Data, No Filtering)', fontsize=14)
    
    # 添加图例
    plt.legend(fontsize=10, loc='best')
    
    plt.grid(True, alpha=0.3)
    
    # 设置合理的y轴范围
    # 找到所有f值的范围
    all_f_values = []
    for _, f_orig in all_curves_data_full:
        all_f_values.extend(f_orig)
    
    if all_f_values:
        f_min = min(all_f_values)
        f_max = max(all_f_values)
        # 设置y轴范围，留10%的边距
        plt.ylim(f_min - 0.1*(f_max-f_min), f_max + 0.1*(f_max-f_min))
    
    # 保存第二个图形
    if save_figure:
        filename2 = '/home/tyt/project/Single-chain/opt+R/Multichain_results/f_r_curves_all_data_gray.png'
        plt.savefig(filename2, dpi=300, bbox_inches='tight')
        print(f"Figure 2 saved as {filename2}")
        print(f"Total curves: {len(all_curves_data_full)}")
        print(f"Total data points: {kept_points_total + filtered_points_total}")

def main():
    # 设置文件路径（请根据实际情况修改）
    r_file_path = '/home/tyt/project/Single-chain/opt+R/Multichain_results/r_values.csv'
    f_file_path = '/home/tyt/project/Single-chain/opt+R/Multichain_results/f_values.csv'
    
    # 设置f的上限值（仅用于可视化）
    f_upper_limit = 15.0  # 可视化时只显示f < 15的数据
    
    # 读取CSV文件
    df_r, df_f = read_csv_files(r_file_path, f_file_path)
    
    # 绘制曲线
    plot_f_r_curves(df_r, df_f, 
                   f_upper_limit=f_upper_limit,
                   title="f vs r Curves", 
                   save_figure=True)
    
    print('\nProcess completed!')
    plt.show()

if __name__ == "__main__":
    main()
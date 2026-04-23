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

def plot_f_r_curves(df1, df2, f_upper_limit=None, title="f-r Curves", save_figure=False):
    """
    绘制f-r曲线
    f_upper_limit: f值的上限，可视化时只显示f < f_upper_limit的数据
    平均值计算使用所有数据，但可视化时平均值也只显示到上限值以下
    """
    if df1 is None or df2 is None:
        print("DataFrames are empty. Cannot plot.")
        return
    
    # 获取列数，确保两个文件列数相同
    num_columns = min(len(df1.columns), len(df2.columns))
    
    # 设置颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, num_columns))
    
    # 用于平均值计算的数据（保留所有数据）
    all_curves_data_full = []
    # 用于可视化的数据（过滤后）
    all_curves_data_filtered = []
    
    # 收集所有r值范围，用于生成统一的插值r数组
    all_r_min = []
    all_r_max = []
    
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
            all_r_min.append(np.min(r_filtered))
            all_r_max.append(np.max(r_filtered))
    
    if len(all_curves_data_filtered) == 0:
        print("No valid data after filtering.")
        return
    
    # ========== 平均值计算部分（使用所有数据） ==========
    # 确定用于平均值计算的r值范围
    # 首先收集所有原始数据的r值范围
    all_r_min_full = []
    all_r_max_full = []
    for r_orig, f_orig in all_curves_data_full:
        all_r_min_full.append(np.min(r_orig))
        all_r_max_full.append(np.max(r_orig))
    
    # 确定用于平均值计算的统一r值范围
    # 使用所有数据的并集范围
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
    if all_r_min and all_r_max:
        common_r_min = 0  # 取所有过滤后曲线r值的最小值中的最大值
        common_r_max = 450  # 取所有过滤后曲线r值的最大值中的最小值
    else:
        # 如果没有过滤后的数据，使用平均值曲线的范围
        common_r_min = r_min_full
        common_r_max = r_max_full
    
    # ========== 可视化部分 ==========
    # 设置图形
    plt.figure(figsize=(10, 6))
    
    # 生成用于可视化的统一r值数组
    r_uniform_viz = np.linspace(common_r_min, common_r_max, 10000)
    
    # 为每条过滤后的曲线绘制插值曲线
    for i, (r_orig, f_orig) in enumerate(all_curves_data_filtered):
        # 线性插值
        interp_func = interpolate.interp1d(r_orig, f_orig, kind='linear', 
                                          bounds_error=False, fill_value=np.nan)
        
        # 在统一r值下计算插值f值
        f_interp = interp_func(r_uniform_viz)
        
        # 绘制插值曲线
        valid_mask = ~np.isnan(f_interp)
        if np.any(valid_mask):
            plt.plot(r_uniform_viz[valid_mask], f_interp[valid_mask], 
                    color=colors[i % len(colors)], linewidth=1.5, alpha=0.7)
    
    # ========== 绘制平均值曲线（只显示到上限值以下） ==========
    # 找到平均值曲线在可视化范围内的部分
    viz_mask_full = (r_uniform_full >= common_r_min) & (r_uniform_full <= common_r_max)
    r_avg_viz = r_uniform_full[viz_mask_full]
    f_avg_viz = f_avg_full[viz_mask_full]
    
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
                plt.plot(segment_r, segment_f, 'b-', linewidth=3.0)
        
        # 只在第一个段添加图例
        if segments:
            plt.plot([], [], 'b-', linewidth=3.0, label='Average (all data)')
    else:
        # 如果没有限制，直接绘制整个平均值曲线
        plt.plot(r_avg_viz, f_avg_viz, 'b-', linewidth=3.0, label='Average (all data)')
    
    # ========== 设置图形属性 ==========
    plt.xlabel('r', fontsize=12)
    plt.ylabel('f', fontsize=12)
    
    # 添加图例
    plt.legend(fontsize=10)
    
    # 添加过滤信息
    if f_upper_limit is not None and filtered_points > 0:
        plt.title(f'{title} (f < {f_upper_limit})', fontsize=14)
        print(f"Visualization: Filtered out {filtered_points} points with f >= {f_upper_limit}")
        print(f"Visualization: Kept {kept_points} points")
        print(f"Average calculation: Used all {kept_points + filtered_points} points")
        print(f"Average curve also truncated where f >= {f_upper_limit}")
    else:
        plt.title(title, fontsize=14)
        print(f"Used all {kept_points} points for both visualization and average")
    
    plt.grid(True, alpha=0.3)
    
    # 设置y轴范围（如果需要）
    if f_upper_limit is not None:
        plt.ylim(0, f_upper_limit * 1.1)  # 留10%的边距
    
    # 保存图形（如果需要）
    if save_figure:
        if f_upper_limit is not None:
            filename = f'/home/tyt/project/Single-chain/opt+R/Rand_L+E/column_format/f_r_curves_interp_filter{f_upper_limit}.png'
        else:
            filename = '/home/tyt/project/Single-chain/opt+R/Rand_L+E/column_format/f_r_curves_interp.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved as {filename}")
        print(f"Visualization r range: [{common_r_min:.4f}, {common_r_max:.4f}]")
        print(f"Average calculation r range: [{r_min_full:.4f}, {r_max_full:.4f}]")
        print(f"Number of interpolation points: {len(r_uniform_viz)}")

def main():
    # 设置文件路径（请根据实际情况修改）
    file1_path = '/home/tyt/project/Single-chain/opt+R/Rand_L+E/column_format/r_values.csv'
    file2_path = '/home/tyt/project/Single-chain/opt+R/Rand_L+E/column_format/f_values.csv'
    
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

if __name__ == "__main__":
    main()
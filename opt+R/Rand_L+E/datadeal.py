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
    f_upper_limit: f值的上限，仅保留f < f_upper_limit的数据
    """
    if df1 is None or df2 is None:
        print("DataFrames are empty. Cannot plot.")
        return
    
    # 获取列数，确保两个文件列数相同
    num_columns = min(len(df1.columns), len(df2.columns))
    
    # 设置图形
    plt.figure(figsize=(10, 6))
    
    # 设置颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, num_columns))
    
    # 统计过滤掉的数据点
    filtered_points = 0
    kept_points = 0
    
    # 收集所有r值范围，用于生成统一的插值r数组
    all_r_min = []
    all_r_max = []
    
    # 为每条曲线收集原始数据（过滤前）
    all_curves_data = []
    for i in range(num_columns):
        r = df1.iloc[:, i].dropna().values
        f = df2.iloc[:, i].dropna().values
        
        # 确保r和f长度相同
        min_len = min(len(r), len(f))
        r = r[:min_len]
        f = f[:min_len]
        
        # 应用f上限过滤
        if f_upper_limit is not None:
            mask = f < f_upper_limit
            filtered_points += sum(~mask)
            kept_points += sum(mask)
            
            r = r[mask]
            f = f[mask]
        
        # 如果过滤后仍有数据，则保存
        if len(r) > 2:  # 至少需要2个点才能进行线性插值
            all_curves_data.append((r, f))
            all_r_min.append(np.min(r))
            all_r_max.append(np.max(r))
    
    # 生成统一的r值数组用于插值
    if len(all_curves_data) == 0:
        print("No valid data after filtering.")
        return
    
    # 确定所有曲线的公共r值范围
    common_r_min = 0  # 取所有曲线r值的最小值中的最大值
    common_r_max = 450 # 取所有曲线r值的最大值中的最小值
    print(common_r_min,common_r_max)
    
    # 如果公共范围无效，使用所有曲线的并集
    if common_r_min > common_r_max:
        print("Warning: No common r range found. Using union of all ranges.")
        common_r_min = min(all_r_min)
        common_r_max = max(all_r_max)
    
    # 生成统一的r值数组（200个点）
    r_uniform = np.linspace(common_r_min, common_r_max, 10000)

    fsum = np.zeros_like(r_uniform)
    count = np.zeros_like(r_uniform)  # 记录有效数据点数量

    # 为每条曲线绘制插值后的曲线
    for i, (r_orig, f_orig) in enumerate(all_curves_data):
        # 线性插值
        interp_func = interpolate.interp1d(r_orig, f_orig, kind='linear', 
                                          bounds_error=False, fill_value=np.nan)
        
        # 在统一r值下计算插值f值
        f_interp = interp_func(r_uniform)
        # 绘制插值曲线
        valid_mask = ~np.isnan(f_interp)
    
        fsum[valid_mask] += f_interp[valid_mask]
        count[valid_mask] += 1  # 只对有效点计数
        if np.any(valid_mask):
            plt.plot(r_uniform[valid_mask], f_interp[valid_mask], 
                    color=colors[i % len(colors)], linewidth=1.5)
    
    # 避免除以0
    with np.errstate(divide='ignore', invalid='ignore'):
        f_avg = np.where(count > 0, fsum / count, np.nan)


    # 绘制平均值曲线（只绘制有数据的地方）
    valid_avg_mask = ~np.isnan(f_avg)
    plt.plot(r_uniform[valid_avg_mask], f_avg[valid_avg_mask], 'b-', linewidth=2.0)
    # 设置图形属性
    plt.xlabel('r', fontsize=12)
    plt.ylabel('f', fontsize=12)
    
    # 添加过滤信息
    if f_upper_limit is not None and filtered_points > 0:
        plt.title(f'{title} (f < {f_upper_limit})', fontsize=14)
        print(f"Filtered out {filtered_points} points with f >= {f_upper_limit}")
        print(f"Kept {kept_points} points after filtering")
    else:
        plt.title(title, fontsize=14)
    
    plt.grid(True, alpha=0.3)
    
    # 保存图形（如果需要）
    if save_figure:
        filename = '/home/tyt/project/Single-chain/opt+R/Rand_L+E/column_format/f_r_curves(interp).png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")
        print(f"Uniform r range: [{common_r_min:.4f}, {common_r_max:.4f}]")
        print(f"Number of interpolation points: {len(r_uniform)}")

def main():
    # 设置文件路径（请根据实际情况修改）
    file1_path = '/home/tyt/project/Single-chain/opt+R/Rand_L+E/column_format/r_values.csv'  # 包含r数据的文件
    file2_path = '/home/tyt/project/Single-chain/opt+R/Rand_L+E/column_format/f_values.csv' # 包含f数据的文件
    
    # 设置f的上限值（根据需要调整）
    f_upper_limit = 10.0  # 例如：只保留f < 10的数据
    
    # 读取CSV文件
    df1, df2 = read_csv_files(file1_path, file2_path)
    
    # 绘制曲线，应用f上限过滤
    plot_f_r_curves(df1, df2, 
                   f_upper_limit=f_upper_limit,
                   title="f vs r Curves(interp)", 
                   save_figure=True)
    
    print('\nProcess completed!')

if __name__ == "__main__":
    main()
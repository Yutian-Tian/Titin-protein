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

def plot_n_r_curves(df1, df2, title="n-r Curves", save_figure=False):
    """
    绘制n-r曲线
    仅读取文件数据，并按照要求进行可视化：
    1. 原始数据用灰色半透明线绘制
    2. 计算平均值并用蓝色实线绘制
    3. 不进行任何过滤或限制
    """
    if df1 is None or df2 is None:
        print("DataFrames are empty. Cannot plot.")
        return
    
    # 获取列数，确保两个文件列数相同
    num_columns = min(len(df1.columns), len(df2.columns))
    
    # 用于平均值计算的数据（所有原始数据）
    all_curves_data = []
    
    # 收集所有r值范围，用于生成统一的插值r数组
    all_r_min = []
    all_r_max = []
    
    # 统计信息
    total_points = 0
    
    # 首先处理所有数据，准备用于平均值计算
    for i in range(num_columns):
        r = df1.iloc[:, i].dropna().values
        n = df2.iloc[:, i].dropna().values
        
        # 确保r和n长度相同
        min_len = min(len(r), len(n))
        r = r[:min_len]
        n = n[:min_len]
        
        # 保存所有数据用于平均值计算
        all_curves_data.append((r.copy(), n.copy()))
        
        # 统计总点数
        total_points += min_len
        
        # 收集r值范围
        if len(r) > 0:
            all_r_min.append(np.min(r))
            all_r_max.append(np.max(r))
    
    if len(all_curves_data) == 0:
        print("No valid data found.")
        return
    
    # ========== 平均值计算部分 ==========
    # 确定用于平均值计算的r值范围
    r_min = min(all_r_min)
    r_max = max(all_r_max)
    
    # 生成用于平均值计算的统一r值数组
    r_uniform = np.linspace(r_min, r_max, 10000)
    nsum = np.zeros_like(r_uniform)
    count = np.zeros_like(r_uniform)
    
    # 对所有数据进行插值并求和
    for r_orig, n_orig in all_curves_data:
        # 线性插值
        interp_func = interpolate.interp1d(r_orig, n_orig, kind='linear', 
                                          bounds_error=False, fill_value=np.nan)
        
        # 在统一r值下计算插值n值
        n_interp = interp_func(r_uniform)
        
        # 累加有效值
        valid_mask = ~np.isnan(n_interp)
        nsum[valid_mask] += n_interp[valid_mask]
        count[valid_mask] += 1
    
    # 计算平均值
    with np.errstate(divide='ignore', invalid='ignore'):
        n_avg = np.where(count > 0, nsum / count, np.nan)
    
    # ========== 设置图形 ==========
    plt.figure(figsize=(10, 6))
    
    # 设置灰线的参数
    GRAY_COLOR = 'gray'
    GRAY_ALPHA = 0.1
    GRAY_LINEWIDTH = 1.0
    
    # 设置平均值曲线的样式
    AVG_COLOR = 'blue'
    AVG_LINEWIDTH = 1.5
    
    # ========== 绘制所有原始曲线 ==========
    for i, (r_orig, n_orig) in enumerate(all_curves_data):
        if len(r_orig) > 2:  # 至少需要3个点才能绘制曲线
            # 线性插值
            interp_func = interpolate.interp1d(r_orig, n_orig, kind='linear', 
                                              bounds_error=False, fill_value=np.nan)
            
            # 生成该曲线的r值网格
            r_curve = np.linspace(np.min(r_orig), np.max(r_orig), 1000)
            n_interp = interp_func(r_curve)
            
            # 绘制插值曲线（半透明灰线）
            valid_mask = ~np.isnan(n_interp)
            if np.any(valid_mask):
                plt.plot(r_curve[valid_mask], n_interp[valid_mask], 
                        color=GRAY_COLOR, linewidth=GRAY_LINEWIDTH, alpha=GRAY_ALPHA)
    
    # ========== 绘制平均值曲线 ==========
    # 绘制完整的平均值曲线
    valid_avg_mask = ~np.isnan(n_avg)
    if np.any(valid_avg_mask):
        plt.plot(r_uniform[valid_avg_mask], n_avg[valid_avg_mask], 
                color=AVG_COLOR, linewidth=AVG_LINEWIDTH, 
                label=f'Average (n={len(all_curves_data)})')
    
    # ========== 设置图形属性 ==========
    plt.xlabel('$r$', fontsize=12)
    plt.ylabel('$n_u$', fontsize=12)
    plt.title(title, fontsize=14)
    
    # 添加图例
    plt.legend(fontsize=10, loc='best')
    
    # 设置刻度朝向内，并显示上轴和右轴
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    
    plt.grid(True, alpha=0.3)
    
    # 设置合理的y轴范围
    # 找到所有n值的范围
    all_n_values = []
    for _, n_orig in all_curves_data:
        all_n_values.extend(n_orig)
    
    if all_n_values:
        n_min = min(all_n_values)
        n_max = max(all_n_values)
        # 设置y轴范围，留10%的边距
        plt.ylim(n_min - 0.1*(n_max-n_min), n_max + 0.1*(n_max-n_min))
    
    # 保存图形
    if save_figure:
        filename = '/home/tyt/project/Single-chain/opt+R/Rand_length/column_format/n_r_curves_all_data.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")
        print(f"Total curves: {len(all_curves_data)}")
        print(f"Total data points: {total_points}")
        print(f"r range: [{r_min:.4f}, {r_max:.4f}]")
    
    plt.show()

def main():
    # 设置文件路径（请根据实际情况修改）
    file1_path = '/home/tyt/project/Single-chain/opt+R/Rand_length/column_format/r_values.csv'
    file2_path = '/home/tyt/project/Single-chain/opt+R/Rand_length/column_format/n_values.csv'
    
    # 读取CSV文件
    df1, df2 = read_csv_files(file1_path, file2_path)
    
    # 绘制曲线（不再有上限限制参数）
    plot_n_r_curves(df1, df2, 
                   title="$n_u$ vs $r$ Curves", 
                   save_figure=True)
    
    print('\nProcess completed!')

if __name__ == "__main__":
    main()
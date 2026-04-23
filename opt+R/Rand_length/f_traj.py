# 用于处理数据，原始轨迹(gray)+平均轨迹(blue)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import os
import warnings

def process_and_plot(f_limit, save_path, ffile_path, rfile_path):
    """
    处理并绘制f和r数据
    
    参数:
    f_limit: f数值的上限
    save_path: 图片保存路径
    """
    
    # 1. 读取数据文件
    try:
        f_data = pd.read_csv(ffile_path)
        r_data = pd.read_csv(rfile_path)
    except FileNotFoundError as e:
        print(f"文件读取错误: {e}")
        return None
    
    # 确保两个文件有相同的列数
    if f_data.shape[1] != r_data.shape[1]:
        print("错误: 两个文件的列数不一致!")
        return None
    
    # 获取列名
    columns = f_data.columns
    num_groups = len(columns)
    
    print(f"找到 {num_groups} 组数据")
    
    # 2. 数据处理和插值准备
    processed_groups = []
    all_r_values = []
    
    for col in columns:
        # 获取当前组的f和r数据
        f_col = f_data[col].values
        r_col = r_data[col].values
        
        # 移除NaN值
        valid_mask = ~np.isnan(f_col) & ~np.isnan(r_col)
        f_valid = f_col[valid_mask]
        r_valid = r_col[valid_mask]
        
        if len(f_valid) < 2:
            continue
        
        # 添加到所有r值集合中
        all_r_values.extend(r_valid.tolist())
        
        processed_groups.append({
            'name': col,
            'f': f_valid,
            'r': r_valid
        })
    
    if not processed_groups:
        print("错误: 没有可处理的数据!")
        return None
    
    # 获取所有组的r值并集，并排序去重
    all_r_unique = np.sort(np.unique(all_r_values))
    print(f"所有组r值的并集范围: [{all_r_unique[0]:.4f}, {all_r_unique[-1]:.4f}], 共 {len(all_r_unique)} 个点")
    
    # 3. 分段线性插值
    interpolated_f_values = []
    valid_groups = []
    
    for group in processed_groups:
        f_vals = group['f']
        r_vals = group['r']
        
        # 确保r值是递增的（插值需要）
        sort_idx = np.argsort(r_vals)
        r_sorted = r_vals[sort_idx]
        f_sorted = f_vals[sort_idx]
        
        # 去除重复的r值
        r_unique, idx_unique = np.unique(r_sorted, return_index=True)
        f_unique = f_sorted[idx_unique]
        
        if len(r_unique) < 2:
            continue
        
        # 创建插值函数
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f_interp_func = interpolate.interp1d(
                    r_unique, 
                    f_unique, 
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
            
            # 在r并集范围内插值
            f_interp = f_interp_func(all_r_unique)
            interpolated_f_values.append(f_interp)
            valid_groups.append(group)
            
        except Exception as e:
            continue
    
    if not interpolated_f_values:
        print("错误: 没有成功插值的数据!")
        return None
    
    # 转换为数组
    interpolated_f_array = np.array(interpolated_f_values)
    
    # 4. 计算平均值fa（忽略NaN值）
    fa = np.nanmean(interpolated_f_array, axis=0)
    
    # 5. 寻找最小的临界点rc（第一个fa>=f_limit的位置）
    rc = None
    valid_mask = None
    
    # 遍历所有点，找到第一个fa>=f_limit的位置
    for i in range(len(fa)):
        if fa[i] >= f_limit:
            rc = all_r_unique[i]
            # 取临界点之前的所有点（包括刚好到达临界点的点）
            valid_mask = np.arange(len(fa)) <= i
            break
    
    # 如果没有找到临界点（所有fa都小于f_limit）
    if rc is None:
        rc = all_r_unique[-1]
        valid_mask = np.ones(len(fa), dtype=bool)
    
    # 获取有效的r和fa
    r_valid = all_r_unique[valid_mask]
    fa_valid = fa[valid_mask]
    
    print(f"最小临界点 rc = {rc:.4f}")
    print(f"有效数据范围: [{r_valid[0]:.4f}, {r_valid[-1]:.4f}], 共 {len(r_valid)} 个点")
    
    # 6. 设置r的显示范围
    r_display_max = min(rc + 50.0, all_r_unique[-1] * 1.1)
    
    # 7. 创建图形
    plt.figure(figsize=(14, 8))
    
    # 绘制每个组的背景线（半透明灰线）- 使用原始数据
    for i, group in enumerate(valid_groups):
        if i < 100:  # 限制背景线数量，避免过于密集
            # 只绘制r <= r_display_max的数据
            mask = group['r'] <= r_display_max
            if np.sum(mask) > 1:  # 至少2个点才能画线
                plt.plot(group['r'][mask], group['f'][mask], 
                        color='gray', 
                        alpha=0.2, 
                        linewidth=0.5,
                        label='_nolegend_')
    
    # 绘制平均线（fa-r），只显示到临界点
    plt.plot(r_valid, fa_valid, 
            color='red', 
            linewidth=3, 
            label=f'Average (f_limit={f_limit})')
    
    # 标记临界点
    plt.axvline(x=rc, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=f'rc = {rc:.2f}')
    
    # 标记f_limit水平线
    plt.axhline(y=f_limit, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label=f'f_limit = {f_limit}')
    
    # 8. 设置图形属性
    plt.xlabel('r', fontsize=14)
    plt.ylabel('f', fontsize=14)
    plt.title(f'f-r Relationship (Average of {len(valid_groups)} groups, f_limit={f_limit})', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 设置坐标轴范围
    plt.xlim(0, r_display_max)
    plt.ylim(0, f_limit * 1.2)
    
    # 9. 保存图片
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"图片已保存到: {save_path}")
    
    # 返回处理后的数据
    return {
        'r_common': all_r_unique,
        'fa': fa,
        'r_valid': r_valid,
        'fa_valid': fa_valid,
        'rc': rc,
        'num_groups': len(valid_groups),
        'r_display_max': r_display_max
    }

def main():
    """
    主函数：设置参数并运行处理程序
    """
    print("=== f-r数据可视化程序 ===")
    print("注意: 只寻找最小的临界点rc作为绘图范围")
    
    # 设置参数
    f_limit = 10.0
    
    rfile_path = '/home/tyt/project/Single-chain/opt+R/Rand_length/column_format/r_values.csv'
    ffile_path = '/home/tyt/project/Single-chain/opt+R/Rand_length/column_format/f_values.csv'

    # 设置保存路径
    save_path = "/home/tyt/project/Single-chain/opt+R/Rand_length/column_format/f_r_curves_all_data_gray.png"
    
    # 运行处理程序
    results = process_and_plot(f_limit, save_path, ffile_path, rfile_path)
    
    if results:
        print(f"\n处理完成!")
        print(f"处理了 {results['num_groups']} 组数据")
        print(f"最小临界点 rc = {results['rc']:.4f}")


if __name__ == "__main__":
    main()
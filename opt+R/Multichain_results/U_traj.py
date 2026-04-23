import numpy as np
import matplotlib.pyplot as plt
import os

def read_lines_to_numpy_arrays(file_path):
    """
    读取txt文件并将每行转换为numpy数组
    """
    arrays = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # 去除换行符和首尾空格
            line = line.strip()
            try:
                # 分割字符串并转换为浮点数
                values = [float(x) for x in line.split()]
                
                # 转换为numpy数组
                arr = np.array(values)
                arrays.append(arr)
                
            except ValueError as e:
                print(f"第{line_num}行转换失败: {e}")
                continue
    
    return arrays

def process_single_row(data_row):
    """
    处理单行数据
    返回：插值后的n和U1
    """
    data = np.cumsum(data_row)
    n_original = np.arange(len(data))  # 原始点索引
    n_interp = np.linspace(0, len(data)-1, 1000)  # 插值点
    
    # 线性插值
    U1 = np.interp(n_interp, n_original, data)
    
    return n_interp, U1

def visualize_single_row(n, U1, row_num, output_file=None, U0=1.0):
    """
    可视化单行数据
    """
    # 计算U0*cos(2*pi*n)
    # 调整周期：假设n的最大值对应2π
    n_max = n[-1] if len(n) > 0 else 1
    cos_term = U0 * np.cos(2 * np.pi * n / n_max)
    
    # 计算差值
    diff = U1 - cos_term
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制三条曲线
    plt.plot(n, U1, 'b-', linewidth=2, label='U1(n)')
    plt.plot(n, diff, 'r-', linewidth=1.5, label='Difference (U1 - U0*cos)')
    
    plt.xlabel('n', fontsize=12)
    plt.ylabel('$U(n)$', fontsize=12)
    plt.title(f'Energy Term - Row {row_num}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {output_file}")
        plt.close()
    else:
        plt.show()
    
    # 返回统计信息
    stats = {
        'row_num': row_num,
        'U1_min': U1.min(),
        'U1_max': U1.max(),
        'U1_mean': U1.mean(),
        'diff_min': diff.min(),
        'diff_max': diff.max(),
        'diff_mean': diff.mean()
    }
    
    return stats


def create_summary_plot(all_data, output_file, U0=1.0):
    """
    创建包含所有行的总结图
    """
    plt.figure(figsize=(15, 10))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_data)))  # 为每行分配不同颜色
    
    for i, data_row in enumerate(all_data):
        # 处理单行数据
        n_interp, U1 = process_single_row(data_row)
        
        # 计算差值
        n_max = n_interp[-1] if len(n_interp) > 0 else 1
        cos_term = U0 * np.cos(2 * np.pi * n_interp / n_max)
        diff = U1 - cos_term
        
        # 绘制差值曲线
        plt.plot(n_interp, diff, color=colors[i], linewidth=1.5)
    
    plt.xlabel('n', fontsize=12)
    plt.ylabel('U1(n) - U0*cos(2πn/N)', fontsize=12)
    plt.title(f'All Rows Comparison (U0={U0})', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n总结图已保存到: {output_file}")
    plt.close()

def main():
    # 参数设置
    U0 = 5.0
    file_path = '/home/tyt/project/Single-chain/opt+R/Multichain_results/energy_eff.txt'
    base_output_path = '/home/tyt/project/Single-chain/opt+R/Multichain_results/U_traj'
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(base_output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=" * 50)
    print("开始处理文件...")
    print("=" * 50)
    
    # 读取数据
    all_data = read_lines_to_numpy_arrays(file_path)
    
    if not all_data:
        print("没有读取到有效数据！")
        return
    
    print(f"\n成功读取 {len(all_data)} 行数据")
    
    
    # 创建总结图
    print("\n" + "=" * 50)
    print("创建总结图...")
    print("=" * 50)
    
    summary_file = f"{base_output_path}_summary.png"
    create_summary_plot(all_data, summary_file, U0)
    

if __name__ == "__main__":
    main()
    print('\nProcess completed!')
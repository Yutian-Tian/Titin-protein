import numpy as np
import matplotlib.pyplot as plt

# 参数设置
Dv = 5.0
DeltaE = 70.0
U0 = 10.0
L = 200.0
DeltaL = 30.0
num_points = 1000
scan_points = 1000

# nf的取值范围
n_min = 0.0
n_max = 5.0

# 函数定义 - 添加数值稳定性处理
def U(n):
    return DeltaE * n - U0 * np.cos(2 * np.pi * n)

def F(n, r, epsilon=1e-10):
    Lc = L - (n_max - n) * DeltaL
    if abs(Lc) < epsilon:  # 避免除零
        return np.inf
    
    x_val = (r - (n_max - n) * Dv) / Lc
    # 限制x_val在合理范围内
    x_val = np.clip(x_val, -1 + epsilon, 1 - epsilon)
    
    denominator = 1 - x_val**2
    denominator = np.maximum(denominator, epsilon)
    
    c_val = 1 / (2 * Lc)
    return np.pi**2 * c_val * denominator + 1 / (np.pi * c_val * denominator) + U(n)

# 计算F对r的精确偏导数 - 添加数值稳定性处理
def exact_partial_F_r(n, r, epsilon=1e-10):
    Lc = L - (n_max - n) * DeltaL
    if abs(Lc) < epsilon:  # 避免除零
        return np.nan
    
    x_val = (r - (n_max - n) * Dv) / Lc
    # 限制x_val在合理范围内
    x_val = np.clip(x_val, -1 + epsilon, 1 - epsilon)
    
    # 处理分母接近零的情况
    denominator = 1 - x_val**2
    if abs(denominator) < epsilon:
        denominator = np.sign(denominator) * epsilon
    
    d_term1_dr = -np.pi**2 * x_val / (Lc**2)
    d_term2_dr = 4 * x_val / (np.pi * denominator)
    
    return d_term1_dr + d_term2_dr

# 生成r的取值范围
r_max = L - n_min * DeltaL - 30.0
r_values = np.linspace(n_max * Dv, r_max, num_points)

# 存储结果
optimal_ns = []
min_F_values = []
partial_F_r_values = []

print("开始均匀扫描优化并计算偏导数...")

# 对每个r值进行均匀扫描优化并计算偏导数
for i, r in enumerate(r_values):
    # 对于给定的r，计算n的实际取值范围
    if DeltaL == 0:
        n_lower_bound = n_min
    else:
        n_lower_bound = max(n_min, n_max - (L - r) / DeltaL)
    
    # 如果下界大于上界，跳过
    if n_lower_bound >= n_max:
        optimal_ns.append(np.nan)
        min_F_values.append(np.nan)
        partial_F_r_values.append(np.nan)
        continue
    
    # 在nf的取值范围内均匀采样
    n_samples = np.linspace(n_lower_bound, n_max, scan_points)
    F_samples = []
    
    # 计算每个采样点的F值
    for n in n_samples:
        try:
            F_val = F(n, r)
            F_samples.append(F_val)
        except:
            F_samples.append(np.inf)
    
    F_samples = np.array(F_samples)
    
    # 找到最小F值对应的n
    if np.all(np.isinf(F_samples)):
        optimal_ns.append(np.nan)
        min_F_values.append(np.nan)
        partial_F_r_values.append(np.nan)
        continue
    
    min_index = np.nanargmin(F_samples)
    min_F = F_samples[min_index]
    optimal_n = n_samples[min_index]
    
    optimal_ns.append(optimal_n)
    min_F_values.append(min_F)
    
    # 使用精确导数计算偏导数
    try:
        dF_dr = exact_partial_F_r(optimal_n, r)
        partial_F_r_values.append(dF_dr)
    except Exception as e:
        partial_F_r_values.append(np.nan)
        if (i + 1) % 100 == 0:  # 减少警告频率
            print(f"警告: 在r={r:.3f}, nf={optimal_n:.6f}处计算偏导数失败: {e}")
    
    # 显示进度 - 减少输出频率
    if (i + 1) % 100 == 0:
        print(f"处理进度: {i + 1}/{num_points}, r={r:.3f}, 最优nf={optimal_n:.6f}, 最小F={min_F:.6f}, ∂F/∂r={partial_F_r_values[-1]:.6f}")

# 转换为numpy数组
optimal_ns = np.array(optimal_ns)
min_F_values = np.array(min_F_values)
partial_F_r_values = np.array(partial_F_r_values)

print("优化和偏导数计算完成，开始绘制图形...")

# 创建图形 - 过滤无效数据
valid_mask = ~np.isnan(min_F_values) & ~np.isinf(min_F_values) & ~np.isnan(partial_F_r_values)

if np.any(valid_mask):
    # 创建图形 - 第一组：原结果
    plt.figure(figsize=(15, 5))

    # F-r 图
    plt.subplot(1, 3, 1)
    plt.plot(r_values[valid_mask], min_F_values[valid_mask] - U(optimal_ns[valid_mask]), 'b-', linewidth=1, label="$F_{WLC}$")
    plt.plot(r_values[valid_mask], min_F_values[valid_mask], 'r-', linewidth=1, label="$F$")
    plt.xlabel('R', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    plt.title('Free Energy vs R', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # nf-r 图
    plt.subplot(1, 3, 2)
    plt.plot(r_values[valid_mask], optimal_ns[valid_mask], 'r-', linewidth=1)
    plt.xlabel('R', fontsize=12)
    plt.ylabel('$n_u$', fontsize=12)
    plt.title('$n_u$ vs R', fontsize=14)
    plt.grid(True, alpha=0.3)

    # ∂F/∂r - r 图
    plt.subplot(1, 3, 3)
    plt.plot(r_values[valid_mask], partial_F_r_values[valid_mask], 'g-', linewidth=2)
    plt.xlabel('R', fontsize=12)
    plt.ylabel('force', fontsize=12)
    plt.title('force vs R', fontsize=14)
    plt.grid(True, alpha=0.3)

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('optimization_with_exact_derivative_results.png', dpi=300)

    # 单独绘制偏导数图
    plt.figure(figsize=(10, 6))
    plt.plot(r_values[valid_mask], partial_F_r_values[valid_mask], 'g-', linewidth=2, label='∂F/∂R')
    plt.xlabel('R', fontsize=14)
    plt.ylabel('force', fontsize=14)
    plt.title('force vs R', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # 添加零线参考
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='零线')

    plt.tight_layout()
    plt.savefig('exact_partial_derivative_F_r.png', dpi=300, bbox_inches='tight')

    print("图形已保存")
else:
    print("警告: 没有有效数据可绘制")

# 保存数据到文件 - 只保存有效数据
if np.any(valid_mask):
    data = np.column_stack((r_values[valid_mask], optimal_ns[valid_mask], min_F_values[valid_mask], partial_F_r_values[valid_mask]))
    np.savetxt('optimization_with_exact_derivative_data.csv', 
               data,
               delimiter=',', 
               header='r,n_opt,F_min,partial_F_r',
               comments='',
               fmt='%.6f')
    
    print("结果已保存:")
    print("  综合图: 'optimization_with_exact_derivative_results.png'")
    print("  偏导数图: 'exact_partial_derivative_F_r.png'")
    print("  数据: 'optimization_with_exact_derivative_data.csv'")
else:
    print("警告: 没有有效数据可保存")

# 显示统计信息
valid_indices = ~np.isnan(partial_F_r_values) & ~np.isinf(partial_F_r_values)
if np.any(valid_indices):
    print(f"\n偏导数统计信息:")
    print(f"有效偏导数值数量: {np.sum(valid_indices)}/{num_points}")
    print(f"∂F/∂r最小值: {np.nanmin(partial_F_r_values[valid_indices]):.6f}")
    print(f"∂F/∂r最大值: {np.nanmax(partial_F_r_values[valid_indices]):.6f}")
    print(f"∂F/∂r平均值: {np.nanmean(partial_F_r_values[valid_indices]):.6f}")
    print(f"∂F/∂r标准差: {np.nanstd(partial_F_r_values[valid_indices]):.6f}")

    # 分析偏导数的符号变化
    positive_derivatives = np.sum(partial_F_r_values[valid_indices] > 0)
    negative_derivatives = np.sum(partial_F_r_values[valid_indices] < 0)
    zero_derivatives = np.sum(np.abs(partial_F_r_values[valid_indices]) < 1e-10)

    print(f"\n偏导数符号分析:")
    print(f"正偏导数(∂F/∂r > 0): {positive_derivatives} 个点")
    print(f"负偏导数(∂F/∂r < 0): {negative_derivatives} 个点")
    print(f"接近零的偏导数: {zero_derivatives} 个点")
else:
    print("警告: 没有有效的偏导数值可分析")
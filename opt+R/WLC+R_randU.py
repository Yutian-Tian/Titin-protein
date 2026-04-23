import numpy as np
import matplotlib.pyplot as plt

# 参数设置
a = 0.0
U0 = 10.0
L = 454.0
xi = 30.0
r_points = 10000
scan_points = 1000

# data =  [0, 5.22034355, 7.0977353, 8.23163261, 9.15335156, 10.0, 10.84664844, 11.76836739, 12.9022647, 14.77965645]
data = [0, 1.88949271, 4.72202512, 7.00538118, 9.06445657, 11.06241343, 13.13007276, 15.43549317, 18.3238436, 23.18323688]
data1 = np.array(data)  # U(n)/Δ 的值
data2 = np.linspace(0, 9, 10)  # n 的值 (0到9)
data3 = np.cumsum(data1)

# n_u的取值范围
n_min = 0.0
n_max = float(len(data)) - 1.0

# 创建线性插值函数
def U1_interp(n):
    """
    线性插值函数，返回在n处的U1(n)值
    """
    return np.interp(n, data2, data3)


# 定义最终的U(n)函数
def U(n):
    """
    U(n) = U1(n) - U0 * cos(2*pi*n)
    """
    return U1_interp(n) - U0 * np.cos(2 * np.pi * n)

def F(n, r):
    c_val = 1 / (2 * (L - (n_max-n) * xi))
    x_val = (r - (n_max - n) * a) / (L - (n_max - n) * xi)
    denominator = 1 - x_val**2
#    denominator = np.maximum(denominator, epsilon)
    return np.pi**2 * c_val * denominator + 1 / (1e-16 + np.pi * c_val * denominator) + U(n)

# 计算F对r的精确偏导数
def exact_partial_F_r(n, r):
    """
    计算在给定(n, r)处F对r的精确偏导数
    ∂F/∂r = ∂/∂r [π²c(1-x²) + 1/(πc(1-x²)) + U]
    """
    # 计算中间变量
    Lc = L - (n_max - n) * xi
    x_val = (r - (n_max - n) * a) / Lc
    
    # 计算导数
    d_term1_dr = - np.pi**2 * x_val / Lc**2
    d_term2_dr = 4 * x_val / (np.pi * (1 - x_val**2)**2)
    
    # U(nf)不依赖于r，其导数为0
    return d_term1_dr + d_term2_dr



# 生成r的取值范围
r_max = L - n_min * xi - 5.0
r_values = np.linspace( n_max * a, r_max, r_points)

# 存储结果
optimal_ns = []
min_F_values = []
partial_F_r_values = []  # 存储偏导数值

print("开始均匀扫描优化并计算偏导数...")

# 对每个r值进行均匀扫描优化并计算偏导数
for i, r in enumerate(r_values):
    # 对于给定的r，计算n的实际取值范围
    n_lower_bound = max(n_min, n_max - (L - r) / xi)
    
    # 如果下界大于上界，跳过
    if n_lower_bound >= n_max:
        optimal_ns.append(np.nan)
        min_F_values.append(np.nan)
        partial_F_r_values.append(np.nan)
        continue
    
    # 在n的取值范围内均匀采样
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
    min_index = np.nanargmin(F_samples)
    min_F = F_samples[min_index]
    optimal_n = n_samples[min_index]
    
    optimal_ns.append(optimal_n)
    min_F_values.append(min_F)
    
    # 使用精确导数计算偏导数
    try:
        dF_dr = exact_partial_F_r(optimal_n, r)
        partial_F_r_values.append(dF_dr)
    except:
        partial_F_r_values.append(np.nan)
        print(f"警告: 在r={r:.3f}, nf={optimal_n:.6f}处计算偏导数失败")
    
    # 显示进度
    if (i + 1) % 1000 == 0:
        print(f"处理进度: {i + 1}/{r_points}, r={r:.3f}, 最优nf={optimal_n:.6f}, 最小F={min_F:.6f}, ∂F/∂r={partial_F_r_values[-1]:.6f}")

# 转换为numpy数组
optimal_ns = np.array(optimal_ns)
min_F_values = np.array(min_F_values)
partial_F_r_values = np.array(partial_F_r_values)
eefactor = r_values/(L-(n_max-optimal_ns)*xi)

print("优化和偏导数计算完成，开始绘制图形...")

# 创建图形 - 第一组：原结果
plt.figure(figsize=(15, 5))

# F-r 图
plt.subplot(1, 3, 1)
plt.plot(r_values, min_F_values - U(optimal_ns), 'b-', linewidth=1, marker='o', markersize=1, label = "$F_{WLC}$")
plt.plot(r_values, min_F_values, 'r-', linewidth=1, marker='o', markersize=1, label = "$F$")
plt.xlabel('R', fontsize=12)
plt.ylabel('Energy', fontsize=12)
plt.title('Free Energy vs R', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

# nf-r 图
plt.subplot(1, 3, 2)
plt.plot(r_values, optimal_ns, 'r-', linewidth=1, marker='s', markersize=1)
plt.xlabel('R', fontsize=12)
plt.ylabel('$n_u$', fontsize=12)
plt.title('$n_u$ vs R', fontsize=14)
plt.grid(True, alpha=0.3)

# ∂F/∂r - r 图
plt.subplot(1, 3, 3)
plt.plot(r_values, partial_F_r_values, 'g-', linewidth=2)
plt.xlabel('R', fontsize=12)
plt.ylabel('force', fontsize=12)
plt.title('force vs R', fontsize=14)
plt.grid(True, alpha=0.3)

# 调整布局并保存
plt.tight_layout()
plt.savefig('/home/tyt/project/Single-chain/opt+R/RandU_results.png', dpi=300)

# 单独绘制偏导数图
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(r_values, partial_F_r_values, 'g-', linewidth=2, label='f=∂F/∂R')
plt.xlabel('R', fontsize=14)
plt.ylabel('force', fontsize=14)
plt.title('force vs R', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)


plt.tight_layout()
plt.savefig('/home/tyt/project/Single-chain/opt+R/RandU_force_r.png', dpi=300, bbox_inches='tight')

# 保存数据到文件
data = np.column_stack((r_values, optimal_ns, min_F_values, partial_F_r_values))
np.savetxt('/home/tyt/project/Single-chain/opt+R/RandU_data.csv', 
           data,
           delimiter=',', 
           header='r,n_opt,F_min,partial_F_r',
           comments='',
           fmt='%.6f')

print("结果已保存!")
# 显示统计信息
valid_indices = ~np.isnan(partial_F_r_values)
print(f"\n偏导数统计信息:")
print(f"有效偏导数值数量: {np.sum(valid_indices)}/{r_points}")
print(f"∂F/∂r最小值: {np.nanmin(partial_F_r_values):.6f}")
print(f"∂F/∂r最大值: {np.nanmax(partial_F_r_values):.6f}")
print(f"∂F/∂r平均值: {np.nanmean(partial_F_r_values):.6f}")
print(f"∂F/∂r标准差: {np.nanstd(partial_F_r_values):.6f}")

# 分析偏导数的符号变化
positive_derivatives = np.sum(partial_F_r_values > 0)
negative_derivatives = np.sum(partial_F_r_values < 0)
zero_derivatives = np.sum(np.abs(partial_F_r_values) < 1e-10)

print(f"\n偏导数符号分析:")
print(f"正偏导数(∂F/∂r > 0): {positive_derivatives} 个点")
print(f"负偏导数(∂F/∂r < 0): {negative_derivatives} 个点")
print(f"接近零的偏导数: {zero_derivatives} 个点")
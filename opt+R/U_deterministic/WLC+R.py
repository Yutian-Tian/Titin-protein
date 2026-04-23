import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 参数设置
xi_f = 20.0
xi_u = 40.0
L = xi_u
DeltaE1 = 3.0
DeltaE2 = 0.0
U0 = 3.0
r_points = 2000
scan_points = 2000
print('线性系数：', DeltaE1)
delta = 1/(2*np.pi)*np.arcsin(- DeltaE1/(4*np.pi*U0))

# n_u的取值范围
n_min = 0.0
n_max = 2.0

# 数值精度控制参数
EPSILON = 1e-14  # 更小的epsilon用于避免除零
X_THRESHOLD = 0.999  # x_val的阈值，避免分母过小

# 函数定义
def U(n):
    n1 = n + delta
    return DeltaE1 * n + DeltaE2 * n1**2 - U0 * np.cos(2 * np.pi * n1)

def F(n, r):
    Lc = L - (n_max - n) * (xi_u - xi_f)
    
    # 检查Lc是否为正，避免负数或零
    if Lc <= 0:
        return np.inf
    
    c_val = 1.0 / (2.0 * Lc)
    x_val = (r ) / Lc
    
    # 确保x_val在合理范围内，避免分母为零
    # 使用更严格的条件限制x_val
    if abs(x_val) >= X_THRESHOLD:
        # 当x_val接近1时，使用近似计算避免数值问题
        x_val = np.sign(x_val) * X_THRESHOLD
    
    denominator = 1.0 - x_val**2
    
    # 保护分母，避免为零
    if denominator <= EPSILON:
        denominator = EPSILON
    
    # 计算两项，注意数值稳定性
    term1 = np.pi**2 * c_val * denominator
    
    # 第二项的分母也需要保护
    term2_denominator = np.pi * c_val * denominator
    if term2_denominator <= EPSILON:
        term2_denominator = EPSILON
    
    term2 = 1.0 / term2_denominator
    
    return term1 + term2 + U(n)

# 计算F对r的精确偏导数（优化版本）
def exact_partial_F_r(n, r):
    """
    计算在给定(n, r)处F对r的精确偏导数
    ∂F/∂r = ∂/∂r [π²c(1-x²) + 1/(πc(1-x²)) + U]
    优化数值精度
    """
    # 计算中间变量
    Lc = L - (n_max - n) * (xi_u - xi_f)
    
    # 检查Lc是否为正
    if Lc <= 0:
        return 0.0
    
    x_val = r  / Lc
    
    # 确保x_val在合理范围内
    if abs(x_val) >= X_THRESHOLD:
        x_val = np.sign(x_val) * X_THRESHOLD
    
    denominator = 1.0 - x_val**2
    
    # 保护分母
    if denominator <= EPSILON:
        denominator = EPSILON
    
    # 计算c值
    c_val = 1.0 / (2.0 * Lc)
    
    # 计算导数项，使用数值更稳定的公式
    # 第一项导数: d(π²c(1-x²))/dr = -2π²c x / Lc
    d_term1_dr = -2.0 * np.pi**2 * c_val * x_val / Lc
    
    # 第二项导数: d(1/(πc(1-x²)))/dr = 2x / (πc Lc (1-x²)²)
    # 使用更稳定的计算方式
    term2_factor = 2.0 * x_val / (np.pi * c_val * Lc)
    d_term2_dr = term2_factor / (denominator**2)
    
    return d_term1_dr + d_term2_dr

# 生成r的取值范围（优化边界检查）
r_max = L
r_min = 0.0
r_values = np.linspace(r_min, 0.8*r_max, r_points)

# 存储结果
optimal_ns = np.full(r_points, np.nan)
min_F_values = np.full(r_points, np.nan)
partial_F_r_values = np.full(r_points, np.nan)

print("开始均匀扫描优化并计算偏导数...")

# 对每个r值进行均匀扫描优化并计算偏导数
for i, r in enumerate(r_values):
    # 对于给定的r，计算n的实际取值范围
    n_lower_bound = max(n_min, n_max - (L - r) / (xi_u - xi_f))
    
    # 如果下界大于上界，跳过
    if n_lower_bound >= n_max:
        continue
    
    # 在n的取值范围内均匀采样
    n_samples = np.linspace(n_lower_bound, n_max, scan_points)
    
    # 使用向量化计算提高效率
    F_samples = np.array([F(n, r) for n in n_samples])
    
    # 找到最小F值对应的n
    valid_mask = np.isfinite(F_samples)
    if not np.any(valid_mask):
        continue
    
    min_index = np.argmin(F_samples[valid_mask])
    # 注意：min_index是有效值中的索引，需要映射回原始索引
    original_indices = np.where(valid_mask)[0]
    min_index_original = original_indices[min_index]
    
    min_F = F_samples[min_index_original]
    optimal_n = n_samples[min_index_original]
    
    optimal_ns[i] = optimal_n
    min_F_values[i] = min_F
    
    # 使用精确导数计算偏导数
    try:
        dF_dr = exact_partial_F_r(optimal_n, r)
        partial_F_r_values[i] = dF_dr
    except Exception as e:
        # 如果计算失败，尝试使用数值导数
        try:
            # 使用中心差分法计算数值导数
            h = 1e-6
            F_plus = F(optimal_n, r + h)
            F_minus = F(optimal_n, r - h)
            if np.isfinite(F_plus) and np.isfinite(F_minus):
                dF_dr = (F_plus - F_minus) / (2 * h)
                partial_F_r_values[i] = dF_dr
            else:
                partial_F_r_values[i] = np.nan
        except:
            partial_F_r_values[i] = np.nan
    
    # 显示进度
    if (i + 1) % 100 == 0:
        print(f"处理进度: {i + 1}/{r_points}, r={r:.3f}, 最优nf={optimal_n:.6f}, 最小F={min_F:.6f}, ∂F/∂r={partial_F_r_values[i]:.6f}")

print("优化和偏导数计算完成，开始绘制图形...")

# 过滤有效数据
valid_mask = np.isfinite(optimal_ns) & np.isfinite(min_F_values) & np.isfinite(partial_F_r_values)
r_valid = r_values[valid_mask]
optimal_ns_valid = optimal_ns[valid_mask]
min_F_values_valid = min_F_values[valid_mask]
partial_F_r_values_valid = partial_F_r_values[valid_mask]

# 计算end-to-end factor
eefactor = r_valid / (L - (n_max - optimal_ns_valid) * (xi_u - xi_f))

# 寻找峰值：在xi_f附近找到最大的偏导数值
xi_f_target = xi_f
# 在xi_f附近寻找峰值，设置一个搜索窗口
search_window = 2.0  # 搜索窗口大小
peak_mask = (r_valid >= xi_f_target - search_window) & (r_valid <= xi_f_target + search_window)

if np.any(peak_mask):
    # 在搜索窗口内找到最大偏导数值
    peak_indices = np.where(peak_mask)[0]
    peak_index_local = np.argmax(partial_F_r_values_valid[peak_mask])
    peak_index_global = peak_indices[peak_index_local]
    
    # 峰值信息
    r_peak = r_valid[peak_index_global]
    f_peak = partial_F_r_values_valid[peak_index_global]
    n_peak = optimal_ns_valid[peak_index_global]
    Lc_peak = L - (n_max - n_peak) * (xi_u - xi_f)
    x_val_peak = r_peak / Lc_peak if Lc_peak > 0 else np.nan
    
    print("\n" + "="*60)
    print(f"在 r = {xi_f} 附近找到峰值:")
    print("="*60)
    print(f"峰值位置 r = {r_peak:.6f}")
    print(f"峰值力 f = {f_peak:.6f}")
    print(f"对应的 n = {n_peak:.6f}")
    print(f"对应的 Lc = {Lc_peak:.6f}")
    print(f"对应的 x = r/Lc = {x_val_peak:.6f}")
    print(f"与目标 xi_f = {xi_f} 的偏差: {abs(r_peak - xi_f):.6f}")
    print("="*60)
    
    # 输出峰值附近5个点的详细信息
    window_size = 5
    start_idx = max(0, peak_index_global - window_size)
    end_idx = min(len(r_valid), peak_index_global + window_size + 1)
    
    print(f"\n峰值附近{r_valid[start_idx]:.3f}到{r_valid[end_idx-1]:.3f}的详细信息:")
    print("r\t\tn\t\tf\t\tLc\t\tx=r/Lc")
    print("-"*80)
    for i in range(start_idx, end_idx):
        r_i = r_valid[i]
        n_i = optimal_ns_valid[i]
        f_i = partial_F_r_values_valid[i]
        Lc_i = L - (n_max - n_i) * (xi_u - xi_f)
        x_i = r_i / Lc_i if Lc_i > 0 else np.nan
        marker = " <-- PEAK" if i == peak_index_global else ""
        print(f"{r_i:.6f}\t{n_i:.6f}\t{f_i:.6f}\t{Lc_i:.6f}\t{x_i:.6f}{marker}")
    
    # 在整个范围内寻找最大值（可能不在xi_f附近）
    global_max_index = np.argmax(partial_F_r_values_valid)
    if global_max_index != peak_index_global:
        print(f"\n注意: 全局最大值在 r = {r_valid[global_max_index]:.6f}, f = {partial_F_r_values_valid[global_max_index]:.6f}")
        print(f"与当前峰值的差异: Δr = {r_valid[global_max_index] - r_peak:.6f}, Δf = {partial_F_r_values_valid[global_max_index] - f_peak:.6f}")
else:
    print(f"\n警告: 在 r = {xi_f} ± {search_window} 范围内未找到有效数据点")
    print(f"r的有效范围: [{min(r_valid):.3f}, {max(r_valid):.3f}]")

# 设置全局字体和样式
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'STIXGeneral']
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.linewidth'] = 0.5

# 颜色方案
colors = {
    'primary': '#2E86AB',      # 深蓝色
    'secondary': '#A23B72',    # 深紫色
    'accent': '#F18F01',       # 橙色
    'neutral': '#4A4A48',      # 深灰色
    'light': '#C5C3C6'         # 浅灰色
}

# 创建图形 - 第一组：原结果
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150, constrained_layout=True)

# 左图：n_u vs r
line1 = ax1.plot(r_valid, optimal_ns_valid, 
                 color=colors['primary'], 
                 linewidth=2.0,
                 marker='o',
                 markersize=4,
                 markerfacecolor='white',
                 markeredgecolor=colors['primary'],
                 markeredgewidth=1.0,
                 label=r'$n_u(r)$')

ax1.set_xlabel(r'$r$', fontsize=13, fontweight='medium')
ax1.set_ylabel(r'$n_u$', fontsize=13, fontweight='medium')
ax1.set_title('Unfolding Number vs Extension', fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.legend(loc='best', fontsize=11)

# 标记峰值位置
if 'r_peak' in locals():
    ax1.axvline(x=xi_f, color='red', linestyle='--', alpha=0.5, label=f'xi_f = {xi_f}')
    ax1.plot(r_peak, n_peak, 'ro', markersize=8, label=f'Peak at r={r_peak:.2f}')
    ax1.legend()

# 美化坐标轴
ax1.spines['top'].set_visible(1.2)
ax1.spines['right'].set_visible(1.2)
ax1.spines['left'].set_linewidth(1.2)
ax1.spines['bottom'].set_linewidth(1.2)
ax1.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=True, right=False)

# 右图：f vs r
line2 = ax2.plot(r_valid, partial_F_r_values_valid,
                 color=colors['secondary'],
                 linewidth=2.2,
                 solid_capstyle='round',
                 label=r'$f(r) = \frac{\partial F}{\partial r}$')

ax2.set_xlabel(r'$r$', fontsize=13, fontweight='medium')
ax2.set_ylabel(r'Force $f$', fontsize=13, fontweight='medium')
ax2.set_title('Force-Extension Relation', fontsize=14, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# 标记峰值位置和xi_f位置
if 'r_peak' in locals():
    ax2.axvline(x=xi_f, color='red', linestyle='--', alpha=0.5, label=f'xi_f = {xi_f}')
    ax2.plot(r_peak, f_peak, 'ro', markersize=10, label=f'Peak: r={r_peak:.2f}, f={f_peak:.2f}')
    # 添加峰值标注
    ax2.annotate(f'Peak\nr={r_peak:.2f}\nf={f_peak:.2f}', 
                xy=(r_peak, f_peak), 
                xytext=(r_peak+1, f_peak*0.9),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, color='red', fontweight='bold')

ax2.legend(loc='best', fontsize=11)

# 美化坐标轴
ax2.spines['top'].set_visible(1.2)
ax2.spines['right'].set_visible(1.2)
ax2.spines['left'].set_linewidth(1.2)
ax2.spines['bottom'].set_linewidth(1.2)
ax2.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=True, right=False)

# 保存第一组图
fig1.savefig('/home/tyt/project/Single-chain/opt+R/U_deterministic/results_WLC.png', 
             dpi=300, 
             bbox_inches='tight', 
             facecolor='white',
             edgecolor='none')


# 打印图表信息
print(f"\n第一张图已保存至: /home/tyt/project/Single-chain/opt+R/U_deterministic/results_WLC.png")
print(f"曲线数据点数量: {len(r_valid)}")
print(f"力值范围: [{min(partial_F_r_values_valid):.3f}, {max(partial_F_r_values_valid):.3f}]")


# 保存数据到文件
data = np.column_stack((r_valid, optimal_ns_valid, min_F_values_valid, partial_F_r_values_valid, eefactor))
np.savetxt('/home/tyt/project/Single-chain/opt+R/U_deterministic/WLC+R_data.csv', 
           data,
           delimiter=',', 
           header='r,n_opt,F_min,partial_F_r,eefactor',
           comments='',
           fmt='%.6f')

print("结果已保存!")

# 显示统计信息
print(f"\n偏导数统计信息:")
print(f"有效偏导数值数量: {len(partial_F_r_values_valid)}/{r_points}")
print(f"∂F/∂r最小值: {np.nanmin(partial_F_r_values_valid):.6f}")
print(f"∂F/∂r最大值: {np.nanmax(partial_F_r_values_valid):.6f}")
print(f"∂F/∂r平均值: {np.nanmean(partial_F_r_values_valid):.6f}")
print(f"∂F/∂r标准差: {np.nanstd(partial_F_r_values_valid):.6f}")


# 检查数值问题
print(f"\n数值稳定性检查:")
print(f"F值中出现inf的次数: {np.sum(~np.isfinite(min_F_values))}")
print(f"n值中出现inf的次数: {np.sum(~np.isfinite(optimal_ns))}")
print(f"偏导数中出现inf的次数: {np.sum(~np.isfinite(partial_F_r_values))}")
print(f"eefactor的范围: [{np.min(eefactor):.6f}, {np.max(eefactor):.6f}]")

# 输出更多关于峰值的信息
if 'r_peak' in locals():
    print(f"\n峰值详细信息:")
    print(f"r_peak = {r_peak:.6f} (与xi_f={xi_f}相差{abs(r_peak-xi_f):.6f})")
    print(f"f_peak = {f_peak:.6f}")
    print(f"n_peak = {n_peak:.6f}")
    print(f"Lc_peak = {Lc_peak:.6f}")
    print(f"x_peak = r_peak/Lc_peak = {x_val_peak:.6f}")
    print(f"此时U(n_peak) = {U(n_peak):.6f}")
    print(f"此时F(n_peak, r_peak) = {F(n_peak, r_peak):.6f}")
    print(f"初始F(0, 0) =",F(0.0, 0.0))
    print(f"末态F(1,rs):",F(1,9.052526))
    print(f"差值:",F(1,9.052526)-F(0.0,0.0))
    
    # 分析峰值前后的变化
    if peak_index_global > 0 and peak_index_global < len(r_valid) - 1:
        print(f"\n峰值前后变化:")
        print(f"f(r-Δr) = {partial_F_r_values_valid[peak_index_global-1]:.6f} (r={r_valid[peak_index_global-1]:.6f})")
        print(f"f(r)     = {f_peak:.6f} (峰值)")
        print(f"f(r+Δr) = {partial_F_r_values_valid[peak_index_global+1]:.6f} (r={r_valid[peak_index_global+1]:.6f})")
        print(f"变化率: 上升 {(f_peak - partial_F_r_values_valid[peak_index_global-1]):.6f}, 下降 {(partial_F_r_values_valid[peak_index_global+1] - f_peak):.6f}")
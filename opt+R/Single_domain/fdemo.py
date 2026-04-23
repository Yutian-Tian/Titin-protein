import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

# ==================== 参数设置 ====================
xi_f = 5.0        # 折叠态特征长度
xi_u = 25.0       # 展开态特征长度
E = 2.5          # 自由能差
beta = 1.0       # β参数
U0 = 0.0         # 周期势参数，设为0简化计算
r_min = 0.0       # 最小端端距离
dr = 0.01          # 距离步长

# 设置公共网格上限为 0.9 * xi_u
grid_max = 0.9 * xi_u

# ==================== 文件路径 ====================
ffile_path = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results/f_values.csv'
rfile_path = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results/r_values.csv'
save_path = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results/compare_f_r_curves_two_methods.png'

# ==================== Step 1: 从2个.csv文件中按列读取数据 ====================
print("Step 1: 从CSV文件中读取数据...")
try:
    # 读取r值和f值，假设第一行是表头
    r_data = pd.read_csv(rfile_path, header=0)  # header=0表示第一行是表头
    f_data = pd.read_csv(ffile_path, header=0)  # header=0表示第一行是表头
    
    # 确保两个文件有相同的列数
    assert r_data.shape[1] == f_data.shape[1], "r和f数据文件的列数不一致"
    
    num_trajectories = r_data.shape[1]
    print(f"  读取到 {num_trajectories} 条轨迹数据")
    print(f"  每条轨迹数据点: {r_data.shape[0]} 个")
    
except FileNotFoundError as e:
    print(f"错误: 找不到文件 - {e}")
    print(f"请检查文件路径:")
    print(f"  f文件: {ffile_path}")
    print(f"  r文件: {rfile_path}")
    exit(1)
except pd.errors.EmptyDataError:
    print("错误: 文件为空")
    exit(1)
except Exception as e:
    print(f"错误: 读取文件时发生错误 - {e}")
    exit(1)

# ==================== Step 2: 计算公共网格 ====================
print("Step 2: 计算公共网格...")

# 创建公共网格 - 使用 0.9 * xi_u
r_common = np.linspace(0, grid_max, 1000)
print(f"\n  公共网格: 从0到{grid_max:.3f} (0.9 * xi_u), 共1000个点")
print(f"  注意: 公共网格上限(0.9*xi_u = {grid_max:.3f})")

# ==================== Step 3: 分段线性插值和平均值计算 ====================
print("\nStep 3: 进行分段线性插值并计算平均值...")

# 存储所有轨迹在公共网格上的插值结果
f_interpolated_all = np.full((len(r_common), num_trajectories), np.nan)

for col in range(num_trajectories):
    # 获取当前轨迹的r和f值
    r_raw = r_data.iloc[:, col].values
    f_raw = f_data.iloc[:, col].values
    
    # 移除NaN值
    valid_mask = ~np.isnan(r_raw) & ~np.isnan(f_raw)
    r_valid = r_raw[valid_mask]
    f_valid = f_raw[valid_mask]
    
    if len(r_valid) < 2:
        print(f"  轨迹 {col+1}: 数据点太少 ({len(r_valid)}个), 跳过")
        continue
    
    # 确保r值是单调递增的
    if not np.all(np.diff(r_valid) >= 0):
        # 按r值排序
        sort_idx = np.argsort(r_valid)
        r_valid = r_valid[sort_idx]
        f_valid = f_valid[sort_idx]
    
    # 创建插值函数
    try:
        # 使用线性插值，对于超出原始数据范围的点，返回NaN
        interp_func = interp1d(r_valid, f_valid, kind='linear', 
                               bounds_error=False, fill_value=np.nan)
        
        # 在公共网格上插值
        f_interpolated = interp_func(r_common)
        f_interpolated_all[:, col] = f_interpolated
        
        # 统计有效点数
        valid_points = np.sum(~np.isnan(f_interpolated))
        
    except Exception as e:
        print(f"  轨迹 {col+1}: 插值失败 - {e}")
        continue

# 计算平均值，忽略NaN
f_avg = np.nanmean(f_interpolated_all, axis=1)
valid_avg_points = np.sum(~np.isnan(f_avg))
print(f"\n  插值完成，平均曲线有 {valid_avg_points} 个有效点")

# 计算在每个r点上有效的轨迹数量
valid_trajectories_per_point = np.sum(~np.isnan(f_interpolated_all), axis=1)
print(f"  在每个r点上，参与平均的有效轨迹数: {np.min(valid_trajectories_per_point)} - {np.max(valid_trajectories_per_point)}")

# 找到数据覆盖结束点（有效轨迹数为0的第一个点）
if np.any(valid_trajectories_per_point == 0):
    data_end_idx = np.where(valid_trajectories_per_point == 0)[0][0]
    data_end_r = r_common[data_end_idx]
    print(f"  数据覆盖结束点: r = {data_end_r:.3f}")
else:
    data_end_r = grid_max
    print(f"  数据覆盖完整网格范围")

# ==================== Step 4: 计算理论解（根据图片中的两种方法） ====================
print("\nStep 4: 计算理论解（根据图片中的两种方法）...")

# WLC自由能函数
def F_WLC(r, Lc):
    """
    计算WLC自由能
    F_WLC(x, L_c) = (π²/(2L_c)) * (1 - x²) + (2L_c/(π(1-x²)))
    其中 x = r/L_c
    """
    if Lc <= 0:
        return np.inf
    
    x = r / Lc
    
    # 确保x在有效范围内 (|x| < 1)
    if abs(x) >= 1 - 1e-10:
        return np.inf
    
    if abs(1 - x**2) < 1e-10:
        return 1e10
    
    term1 = (np.pi**2) / (2 * Lc) * (1 - x**2)
    term2 = (2 * Lc) / (np.pi * (1 - x**2))
    
    return term1 + term2

# WLC力函数
def f_WLC(r, Lc):
    """
    计算WLC力
    f = - (π² * x) / Lc² + (4 * x) / (π * (1 - x²)²)
    其中 x = r/Lc
    """
    if Lc <= 0:
        return 0.0
    
    x = r / Lc
    
    # 确保x在有效范围内 (|x| < 1)
    if abs(x) >= 1 - 1e-10:
        # 如果x接近1，返回一个很大的力
        return 1e10 if x > 0 else -1e10
    
    if abs(1 - x**2) < 1e-10:
        return 1e10 if x > 0 else -1e10
    
    term1 = - (np.pi**2 * x) / (Lc**2)
    term2 = (4 * x) / (np.pi * (1 - x**2)**2)
    
    return term1 + term2

def calculate_theory_method1(r_common):
    """
    方法1: f(r) = f_WLC[r, ⟨n⟩(r)]
    其中 ⟨n⟩(r) = exp[-βΔ(r)] / (1 + exp[-βΔ(r)])
    Δ(r) = F_WLC(r, xi_u) - F_WLC(r, xi_f) + ΔE
    
    注意：这里的轮廓长度Lc = ξ_f + ⟨n⟩(ξ_u - ξ_f)
    """
    print("  计算方法1: f(r) = f_WLC[r, ⟨n⟩(r)] ...")
    f_theory1 = np.zeros_like(r_common)
    
    for i, r in enumerate(r_common):
        if r < 1e-10:
            f_theory1[i] = 0.0
            continue
        
        # 计算Δ(r)
        F_folded = F_WLC(r, xi_f)   # F_WLC(r, 0) 折叠状态
        F_unfolded = F_WLC(r, xi_u) # F_WLC(r, 1) 展开状态
        
        # 检查自由能是否有效
        if np.isinf(F_folded) or np.isinf(F_unfolded):
            f_theory1[i] = np.nan
            continue
        
        Delta = F_unfolded - F_folded + E
        
        # 计算⟨n⟩(r)
        exponent = -beta * Delta
        if exponent > 700:
            n_avg = 1.0
        elif exponent < -700:
            n_avg = 0.0
        else:
            n_avg = np.exp(exponent) / (1 + np.exp(exponent))
        
        # 计算轮廓长度
        Lc = xi_f + n_avg * (xi_u - xi_f)
        
        # 计算力
        f_theory1[i] = f_WLC(r, Lc)
    
    print(f"  方法1计算完成")
    return f_theory1

def calculate_theory_method2(r_common):
    """
    方法2: ⟨f⟩(r) = f_WLC(r, 0) p(n=0; r) + f_WLC(r, 1) p(n=1; r)
    其中:
    p(n=0; r) = 1 / (1 + exp[-βΔ(r)])
    p(n=1; r) = exp[-βΔ(r)] / (1 + exp[-βΔ(r)])
    Δ(r) = F_WLC(r, xi_u) - F_WLC(r, xi_f) + ΔE
    """
    print("  计算方法2: ⟨f⟩(r) = f_WLC(r, 0) p(n=0; r) + f_WLC(r, 1) p(n=1; r) ...")
    f_theory2 = np.zeros_like(r_common)
    
    for i, r in enumerate(r_common):
        if r < 1e-10:
            f_theory2[i] = 0.0
            continue
        
        # 计算Δ(r)
        F_folded = F_WLC(r, xi_f)   # F_WLC(r, 0) 折叠状态
        F_unfolded = F_WLC(r, xi_u) # F_WLC(r, 1) 展开状态
        
        # 检查自由能是否有效
        if np.isinf(F_folded) or np.isinf(F_unfolded):
            f_theory2[i] = np.nan
            continue
        
        Delta = F_unfolded - F_folded + E
        
        # 计算概率
        exponent = -beta * Delta
        if exponent > 700:
            p0 = 0.0  # p(n=0; r)
            p1 = 1.0  # p(n=1; r)
        elif exponent < -700:
            p0 = 1.0
            p1 = 0.0
        else:
            exp_term = np.exp(exponent)
            p0 = 1.0 / (1.0 + exp_term)
            p1 = exp_term / (1.0 + exp_term)
        
        # 计算两种状态下的力
        f_folded = f_WLC(r, xi_f)    # f_WLC(r, 0)
        f_unfolded = f_WLC(r, xi_u)  # f_WLC(r, 1)
        
        # 计算平均力
        f_theory2[i] = f_folded * p0 + f_unfolded * p1
    
    print(f"  方法2计算完成")
    return f_theory2

# 计算两种方法的理论曲线
f_theory1 = calculate_theory_method1(r_common)
f_theory2 = calculate_theory_method2(r_common)

# 清理理论数据（插值填补NaN值）
def clean_data(x, y):
    """清理数据，对NaN值进行线性插值"""
    valid_idx = np.where(np.isfinite(y))[0]
    
    if len(valid_idx) < 2:
        return x, y
    
    y_clean = np.copy(y)
    
    for i in range(len(y)):
        if not np.isfinite(y[i]):
            left_idx = valid_idx[valid_idx < i]
            right_idx = valid_idx[valid_idx > i]
            
            if len(left_idx) > 0 and len(right_idx) > 0:
                left = left_idx[-1]
                right = right_idx[0]
                y_clean[i] = y[left] + (y[right] - y[left]) * (x[i] - x[left]) / (x[right] - x[left])
            elif len(left_idx) > 0:
                y_clean[i] = y[left_idx[-1]]
            elif len(right_idx) > 0:
                y_clean[i] = y[right_idx[0]]
    
    return x, y_clean

r_clean1, f_theory1_clean = clean_data(r_common, f_theory1)
r_clean2, f_theory2_clean = clean_data(r_common, f_theory2)

# 计算两种方法之间的差异
diff_12 = np.abs(f_theory1_clean - f_theory2_clean)
avg_diff = np.nanmean(diff_12)
max_diff = np.nanmax(diff_12)
print(f"\n  两种理论方法比较:")
print(f"  平均差异: {avg_diff:.6f}")
print(f"  最大差异: {max_diff:.6f}")

# ==================== Step 5: 可视化 ====================
print("\nStep 5: 绘制可视化图形...")

plt.figure(figsize=(14, 9))

# 绘制所有原始轨迹（半透明灰细线）
for col in range(num_trajectories):
    # 获取原始数据，移除NaN
    r_raw = r_data.iloc[:, col].values
    f_raw = f_data.iloc[:, col].values
    
    valid_mask = ~np.isnan(r_raw) & ~np.isnan(f_raw)
    if np.sum(valid_mask) > 1:
        plt.plot(r_raw[valid_mask], f_raw[valid_mask], 
                color='gray', alpha=0.15, linewidth=0.8, zorder=1)

# 绘制平均曲线fa-r（红色）
plt.plot(r_common, f_avg, color='red', linewidth=3.5, 
         label='Simulation Mean', zorder=3)

# 绘制两种理论曲线
plt.plot(r_clean1, f_theory1_clean, color='green', linewidth=3.0, 
         linestyle='-', label='Theory Method 1: f(r) = f_WLC[r, ⟨n⟩(r)]', zorder=2)

plt.plot(r_clean2, f_theory2_clean, color='blue', linewidth=3.0, 
         linestyle='--', label='Theory Method 2: ⟨f⟩(r) = f_WLC(r,0)p(0) + f_WLC(r,1)p(1)', zorder=2)

# 设置图形属性
plt.xlabel('End-to-end distance $r$', fontsize=18)
plt.ylabel('Average Force $f_a$', fontsize=18)
plt.title('Force-extension Curves: Simulation vs Two Theoretical Methods', fontsize=20, fontweight='bold')
plt.legend(fontsize=14, loc='best')
plt.xlim(0, grid_max)

# 设置刻度标签大小
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()

# 保存图形
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"主图形已保存至: {save_path}")

# ==================== 创建第二个图形：显示有效轨迹数量 ====================
plt.figure(figsize=(12, 6))
plt.plot(r_common, valid_trajectories_per_point, color='blue', linewidth=2)
plt.xlabel('End-to-end distance $r$', fontsize=16)
plt.ylabel('Number of valid trajectories', fontsize=16)
plt.title('Number of valid trajectories at each r point', fontsize=18)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim(0, grid_max)
plt.axvline(x=data_end_r, color='red', linestyle='--', alpha=0.7, label=f'Data end: r={data_end_r:.2f}')
plt.legend(fontsize=12)
plt.tight_layout()

valid_traj_save_path = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results/valid_trajectories_count.png'
plt.savefig(valid_traj_save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"有效轨迹数量图已保存至: {valid_traj_save_path}")

# ==================== 输出总结信息 ====================
print("\n" + "="*60)
print("处理完成！")
print("="*60)
print(f"轨迹数量: {num_trajectories}")
print(f"每条轨迹数据点: {r_data.shape[0]}")
print(f"公共网格范围: 0 到 {grid_max:.3f} (0.9 * ξ_u)")
print(f"方法1理论曲线有效点数: {np.sum(~np.isnan(f_theory1_clean))}")
print(f"方法2理论曲线有效点数: {np.sum(~np.isnan(f_theory2_clean))}")
print(f"模拟平均曲线有效点数: {valid_avg_points}")
print(f"在每个r点上，有效轨迹数量范围: {np.min(valid_trajectories_per_point)} - {np.max(valid_trajectories_per_point)}")
print(f"数据覆盖结束点: r = {data_end_r:.3f}")
print(f"两种理论方法的平均差异: {avg_diff:.6f}")
print(f"两种理论方法的最大差异: {max_diff:.6f}")

# 保存数据到CSV文件
result_data = pd.DataFrame({
    'r': r_common,
    'f_simulation_mean': f_avg,
    'f_theory_method1': f_theory1_clean,
    'f_theory_method2': f_theory2_clean,
    'valid_trajectories_count': valid_trajectories_per_point,
    'difference_method1_method2': diff_12
})
result_csv_path = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results/f_r_curves_comparison_two_methods.csv'
result_data.to_csv(result_csv_path, index=False)
print(f"\n曲线数据已保存至: {result_csv_path}")
print("="*60)
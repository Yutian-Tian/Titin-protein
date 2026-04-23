import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

# ==================== 参数设置 ====================
xi_f = 10.0        # 折叠态特征长度
xi_u = 20.0       # 展开态特征长度
E = 10.0          # 自由能差
r_min = 0.0       # 最小端端距离
dr = 0.01          # 距离步长
f_limit = 30.0    # 力的上限值，只显示小于此值的数据

# ==================== 文件路径 ====================
ffile_path = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results/f_values.csv'
rfile_path = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results/r_values.csv'
save_path = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results/compare_f_r_curves.png'

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

# ==================== Step 2: 计算基于数据的公共网格 ====================
print("Step 2: 计算基于数据的公共网格...")

# 计算所有轨迹中r的最大值（忽略NaN）
max_r_all = -np.inf
for col in range(num_trajectories):
    r_raw = r_data.iloc[:, col].values
    r_valid = r_raw[~np.isnan(r_raw)]
    if len(r_valid) > 0:
        max_r_current = np.max(r_valid)
        if max_r_current > max_r_all:
            max_r_all = max_r_current

print(f"  所有轨迹中r的最大值: {max_r_all:.3f}")

# 设置基于数据的公共网格上限为最大r值的0.9倍
data_grid_max = 0.9 * max_r_all

# 创建基于数据的公共网格
r_data_common = np.linspace(0, data_grid_max, 1000)
print(f"\n  基于数据的公共网格: 从0到{data_grid_max:.3f} (0.9 * max_r), 共1000个点")

# ==================== Step 3: 理论解的计算网格 ====================
print("Step 3: 设置理论解的计算网格...")

# 设置理论解的计算范围为 [0, 0.9 * xi_u]
theory_grid_max = 0.9 * xi_u

# 创建理论解的计算网格
r_theory_common = np.linspace(0, theory_grid_max, 1000)
print(f"  理论解计算网格: 从0到{theory_grid_max:.3f} (0.9 * xi_u), 共1000个点")

# ==================== Step 4: 分段线性插值和平均值计算 ====================
print("\nStep 4: 进行分段线性插值并计算平均值...")
print(f"  只使用f < {f_limit}的数据点...")

# 存储所有轨迹在基于数据的公共网格上的插值结果
f_interpolated_all = np.full((len(r_data_common), num_trajectories), np.nan)

for col in range(num_trajectories):
    # 获取当前轨迹的r和f值
    r_raw = r_data.iloc[:, col].values
    f_raw = f_data.iloc[:, col].values
    
    # 移除NaN值
    valid_mask = ~np.isnan(r_raw) & ~np.isnan(f_raw)
    r_valid = r_raw[valid_mask]
    f_valid = f_raw[valid_mask]
    
    # 只保留f值小于f_limit的数据点
    f_limit_mask = f_valid < f_limit
    r_valid = r_valid[f_limit_mask]
    f_valid = f_valid[f_limit_mask]
    
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
        
        # 在基于数据的公共网格上插值
        f_interpolated = interp_func(r_data_common)
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
    data_end_r = r_data_common[data_end_idx]
    print(f"  数据覆盖结束点: r = {data_end_r:.3f}")
else:
    data_end_r = data_grid_max
    print(f"  数据覆盖完整网格范围")

# ==================== Step 5: 计算理论解 ====================
print("\nStep 5: 计算理论解...")

def p_f(f):
    """数值稳定的展开概率计算"""
    term = -E + 0.5*f * (xi_u - xi_f)
    
    if term > 700.0:
        return 0.0
    elif term < -700.0:
        return 1.0
    else:
        return 1.0 / (1.0 + np.exp(term))

def equations(vars, r):
    """定义方程组：求解力f和展开概率pf"""
    f = vars[0]
    
    # 计算展开概率和轮廓长度
    pf = p_f(f)
    lc = xi_u - pf * (xi_u - xi_f)
    
    # 检查轮廓长度是否有效
    if lc <= 0:
        return [1e10]
    
    # 计算端端比
    x = r / lc
    
    # 检查x是否在有效范围内
    if x >= 1 or abs(1 - x**2) < 1e-10:
        return [1e10]
    
    # 力方程：f - rhs = 0
    rhs = - (np.pi**2 * x) / (lc**2) + (4 * x) / (np.pi * (1 - x**2))
    
    return [f - rhs]

# 为理论网格中的每个r值计算理论f值
f_theory = np.full_like(r_theory_common, np.nan)  # 初始化为NaN

f_guess = 0.0
maxfev = 1000

print("  开始求解理论曲线...")
solved_points = 0
for i, r in enumerate(r_theory_common):
    if r < r_min:
        continue  # 跳过小于r_min的点
    
    # 使用前一个解作为初始猜测
    if i > 0 and not np.isnan(f_theory[i-1]):
        f_guess = f_theory[i-1]
    else:
        f_guess = 0.0
    
    # 使用fsolve求解
    try:
        f_solution = fsolve(equations, f_guess, args=(r,), maxfev=maxfev, full_output=False)
        
        if not np.isfinite(f_solution[0]):
            continue
        
        f_val = f_solution[0]
        
        # 检查解是否在物理合理范围内
        if abs(f_val) > 1000:
            continue
        # 只保留小于f_limit的理论值
        elif f_val < f_limit:
            f_theory[i] = f_val
            solved_points += 1
        else:
            # 如果f_val >= f_limit，设置为NaN
            f_theory[i] = np.nan
            
    except Exception as e:
        # 跳过求解失败的点
        continue

print(f"  理论曲线求解完成，共求解 {solved_points} 个点")

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

r_theory_clean, f_theory_clean = clean_data(r_theory_common, f_theory)
print("  理论曲线清理完成")

# ==================== Step 6: 可视化 ====================
print("\nStep 6: 绘制可视化图形...")

plt.figure(figsize=(14, 9))

# 确定x轴显示范围 - 取理论和数据网格的较大值
x_max = max(theory_grid_max, data_grid_max)

# 绘制所有原始轨迹（半透明灰细线）- 只显示f < f_limit的部分
for col in range(num_trajectories):
    # 获取原始数据，移除NaN
    r_raw = r_data.iloc[:, col].values
    f_raw = f_data.iloc[:, col].values
    
    # 只显示f < f_limit的数据点
    valid_mask = ~np.isnan(r_raw) & ~np.isnan(f_raw) & (f_raw < f_limit)
    if np.sum(valid_mask) > 1:
        plt.plot(r_raw[valid_mask], f_raw[valid_mask], 
                color='gray', alpha=0.15, linewidth=0.8, zorder=1)

# 绘制平均曲线fa-r（红色）- 只显示有效数据点
valid_avg_mask = ~np.isnan(f_avg) & (f_avg < f_limit)
plt.plot(r_data_common[valid_avg_mask], f_avg[valid_avg_mask], color='red', linewidth=3.5, 
         label='Simulation Mean', zorder=3)

# 绘制理论曲线fa-r（绿色）- 只显示有效数据点
valid_theory_mask = ~np.isnan(f_theory_clean) & (f_theory_clean < f_limit)
plt.plot(r_theory_clean[valid_theory_mask], f_theory_clean[valid_theory_mask], color='green', linewidth=3.0, 
         linestyle='--', label='Theoretical Solution', zorder=2)

# 设置图形属性
plt.xlabel('End-to-end distance $r$', fontsize=18)
plt.ylabel('Average Force $f_a$', fontsize=18)
plt.title(f'Force-extension Curves: Simulation vs Theory', fontsize=20, fontweight='bold')
plt.legend(fontsize=16, loc='best')
# plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim(0, x_max)
plt.ylim(0, f_limit * 1.05)  # 设置y轴上限略高于f_limit

# 添加垂直线标记理论网格和数据网格的边界
plt.axvline(x=theory_grid_max, color='green', linestyle=':', alpha=0.5, linewidth=1.5, label='Theoretical grid boundary')
plt.axvline(x=data_grid_max, color='red', linestyle=':', alpha=0.5, linewidth=1.5, label='Data grid boundary')

# 设置刻度标签大小
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()

# 保存图形
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"主图形已保存至: {save_path}")

# ==================== 创建第二个图形：显示有效轨迹数量 ====================
plt.figure(figsize=(12, 6))
plt.plot(r_data_common, valid_trajectories_per_point, color='blue', linewidth=2)
plt.xlabel('End-to-end distance $r$', fontsize=16)
plt.ylabel('Number of valid trajectories', fontsize=16)
plt.title(f'Number of valid trajectories at each r point (f < {f_limit})', fontsize=18)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim(0, data_grid_max)
plt.axvline(x=data_end_r, color='red', linestyle='--', alpha=0.7, label=f'Data end: r={data_end_r:.2f}')
plt.axvline(x=data_grid_max, color='black', linestyle=':', alpha=0.7, label=f'Data grid: r={data_grid_max:.2f}')
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
print(f"所有轨迹中r的最大值: {max_r_all:.3f}")
print(f"基于数据的公共网格范围: 0 到 {data_grid_max:.3f} (0.9 * max_r)")
print(f"理论解计算网格范围: 0 到 {theory_grid_max:.3f} (0.9 * ξ_u)")
print(f"力的上限值: f_limit = {f_limit}")
print(f"理论曲线有效点数: {np.sum(~np.isnan(f_theory_clean))}")
print(f"模拟平均曲线有效点数: {valid_avg_points}")
print(f"在每个r点上，有效轨迹数量范围: {np.min(valid_trajectories_per_point)} - {np.max(valid_trajectories_per_point)}")
print(f"数据覆盖结束点: r = {data_end_r:.3f}")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ==================== 参数设置 ====================
xi_f = 10.0        # 折叠态特征长度
xi_u = 20.0       # 展开态特征长度
E = 26          # 自由能差（未给出，假设为0）
r_min = 0.1       # 最小端端距离
r_max = 0.95*xi_u    # 最大端端距离（小于 xi_u=30）
rs = 9.06
dr = 0.1          # 距离步长

# ==================== 定义函数（数值稳定版本） ====================
def p_f(f):
    """数值稳定的展开概率计算"""
    # 原始公式: p_f(f) = 1 / (1 + exp(-E + f*rs))
    term = -E + f * rs
    
    # 使用数值稳定的sigmoid函数
    # 当term很大时，使用近似值
    if term > 100.0:
        return 0.0
    elif term < -100.0:
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
        # 返回大值，使求解器远离无效区域
        return [1e10]
    
    # 计算端端比
    x = r / lc
    
    # 检查x是否在有效范围内
    if x >= 1 or abs(1 - x**2) < 1e-10:
        # 返回大值，使求解器远离无效区域
        return [1e10]
    
    # 力方程：f - rhs = 0
    rhs = - (np.pi**2 * x) / (lc**2) + (4 * x) / (np.pi * (1 - x**2)**2)
    
    return [f - rhs]

# ==================== 使用fsolve求解 ====================
r_vals = np.arange(r_min, r_max + dr, dr)  # r 数组
f_vals = []     # 存储力
p_u_vals = []   # 存储未展开概率

# 初始猜测值
f_guess = 0.0

# 设置求解器参数
maxfev = 1000  # 最大函数调用次数

print("开始求解...")

for i, r in enumerate(r_vals):
    # 使用前一个解作为初始猜测（连续性假设）
    if i > 0 and len(f_vals) > 0:
        f_guess = f_vals[-1]
    else:
        f_guess = 0.0
    
    # 使用fsolve求解
    try:
        # fsolve求解单个方程
        f_solution = fsolve(equations, f_guess, args=(r,), maxfev=maxfev, full_output=False)
        
        # 检查解是否合理
        if not np.isfinite(f_solution[0]):
            f_vals.append(np.nan)
            p_u_vals.append(np.nan)
            print(f"警告: r={r:.2f} 处得到非有限解，跳过")
            continue
        
        f_val = f_solution[0]
        
        # 计算对应的未展开概率
        pf_val = p_f(f_val)
        p_u_val = 1.0 - pf_val
        
        # 检查解是否在物理合理范围内
        if abs(f_val) > 1000 or not np.isfinite(p_u_val):
            f_vals.append(np.nan)
            p_u_vals.append(np.nan)
            print(f"警告: r={r:.2f} 处的解超出物理范围，跳过")
        else:
            f_vals.append(f_val)
            p_u_vals.append(p_u_val)
            
    except Exception as e:
        print(f"r={r:.2f}: 求解失败 - {e}")
        f_vals.append(np.nan)
        p_u_vals.append(np.nan)

# 转换为numpy数组
f_vals = np.array(f_vals)
p_u_vals = np.array(p_u_vals)

# ==================== 清理数据（插值填补NaN值） ====================
def clean_data(x, y):
    """清理数据，对NaN值进行线性插值"""
    # 找出有效值索引
    valid_idx = np.where(np.isfinite(y))[0]
    
    if len(valid_idx) < 2:
        return x, y
    
    # 创建新的y值，对NaN进行插值
    y_clean = np.copy(y)
    
    # 对NaN值进行线性插值
    for i in range(len(y)):
        if not np.isfinite(y[i]):
            # 找到前后最近的有效值
            left_idx = valid_idx[valid_idx < i]
            right_idx = valid_idx[valid_idx > i]
            
            if len(left_idx) > 0 and len(right_idx) > 0:
                # 两边都有有效值，线性插值
                left = left_idx[-1]
                right = right_idx[0]
                y_clean[i] = y[left] + (y[right] - y[left]) * (x[i] - x[left]) / (x[right] - x[left])
            elif len(left_idx) > 0:
                # 只有左边有有效值，使用左边值
                y_clean[i] = y[left_idx[-1]]
            elif len(right_idx) > 0:
                # 只有右边有有效值，使用右边值
                y_clean[i] = y[right_idx[0]]
    
    return x, y_clean

# 清理力数据
r_clean, f_clean = clean_data(r_vals, f_vals)

# 清理概率数据
_, p_u_clean = clean_data(r_vals, p_u_vals)

# ==================== 分别绘制曲线 ====================
# 3. 绘制组合对比图（上下排列）
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# 上子图：力曲线
axes[0].plot(r_clean, f_clean, color='tab:red', linewidth=2.5)
axes[0].set_xlabel('end-to-end distance $r$', fontsize=12)
axes[0].set_ylabel('force $f$', fontsize=12)
axes[0].set_title(f'force-extension $f(r)$', fontsize=14)
axes[0].grid(True, alpha=0.3, linestyle='--')

# 下子图：未展开概率曲线
axes[1].plot(r_clean, p_u_clean, color='tab:blue', linewidth=2.5, linestyle='-')
axes[1].set_xlabel('end-to-end distance $r$', fontsize=12)
axes[1].set_ylabel('Probabi;ity $p_u$', fontsize=12)
axes[1].set_title(f'$p_u$ versus $r$', fontsize=14)
axes[1].set_ylim(-0.05, 1.1)
axes[1].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()

# 保存组合对比图
combined_save_path = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results/protein_unfolding_analysis.png'
plt.savefig(combined_save_path, dpi=300, bbox_inches='tight')
print(f"组合对比图已保存至: {combined_save_path}")
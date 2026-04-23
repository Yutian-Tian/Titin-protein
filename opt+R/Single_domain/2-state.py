import numpy as np
import matplotlib.pyplot as plt

# 参数设置
xi_f = 3.0
xi_u = 30.0
DeltaE = 3.0
beta = 1.0
L_c0 = xi_f  # 折叠态轮廓长度
L_c1 = xi_u  # 去折叠态轮廓长度

# 创建r数组，避免接近L_c0=3时发散
r = np.linspace(0.0, 0.8*xi_u, 500)

# WLC自由能函数
def F_WLC(r, L_c):
    x = r / L_c
    # 防止x过于接近1导致分母为零（通过clip限制）
    x = np.clip(x, 0, 0.999)
    term1 = (np.pi**2 / (2 * L_c)) * (1 - x**2)
    term2 = (2 * L_c) / (np.pi * (1 - x**2))
    return term1 + term2

# WLC力函数（未除以持久长度）
def f_WLC(r, L_c):
    x = r / L_c
    x = np.clip(x, 0, 0.999)
    term1 = -np.pi**2 * x / (L_c**2)
    term2 = 4 * x / (np.pi * (1 - x**2)**2)
    return term1 + term2

# 计算各量
x0 = r / L_c0
x1 = r / L_c1

F0 = (1/xi_f) * F_WLC(r, L_c0)
F1 = (1/xi_u) * F_WLC(r, L_c1)

Delta = F1 - F0 + DeltaE
p_u = 1 / (1 + np.exp(Delta))  # 去折叠概率
p_f = 1 - p_u                  # 折叠概率

# 两个状态的力（已除以各自持久长度）
f0 = (1/xi_f) * f_WLC(r, L_c0)
f1 = (1/xi_u) * f_WLC(r, L_c1)

# 方法2：平均力
f_avg = f0 * p_f + f1 * p_u

# 方法1：基于平均<n>的直接力
n_avg = p_u  # <n> = p_u
L_c_avg = L_c0 + n_avg * (L_c1 - L_c0)  # 线性插值轮廓长度
x_avg = r / L_c_avg
x_avg = np.clip(x_avg, 0, 0.999)
f_direct = f_WLC(r, L_c_avg)  # 注意：这里没有额外除以持久长度
# 可选：考虑平均持久长度的缩放
xi_avg = xi_f + n_avg * (xi_u - xi_f)
f_direct_scaled = f_direct / xi_avg  # 缩放版本

# 绘图
plt.figure(figsize=(12, 8))

# 绘制概率
plt.subplot(2, 2, 1)
plt.plot(r, p_u, label='$p_u(r)$', color='blue')
plt.plot(r, p_f, label='$p_f(r)$', color='red')
plt.xlabel('Extension $r$')
plt.ylabel('Probability')
plt.title('Unfolding and folding probability')
plt.legend()
plt.grid(True)

# contour length
plt.subplot(2, 2, 2)
plt.plot(r, L_c_avg, label='Average contour length', color='red')
plt.xlabel('Extension $r$')
plt.ylabel('Contour length $L_c$')
plt.title('Contour length vs Extension')
plt.legend()
plt.grid(True)

# 绘制两种方法得到的力
plt.subplot(2, 2, 3)
plt.plot(r, f_direct, label='Method 1: $f_{WLC}[r, \\langle n \\rangle]$', color='green')
plt.plot(r, f_avg, label='Method 2: $\\langle f \\rangle (r)$', color='orange')
plt.xlabel('Extension $r$')
plt.ylabel('Force')
plt.title('Comparison of two methods for force')
plt.legend()
plt.grid(True)

# 绘制方法1缩放版本与方法2的比较
plt.subplot(2, 2, 4)
plt.plot(r, f_direct_scaled, label='Method 1 scaled by $\\xi_{avg}$', color='purple')
plt.plot(r, f_avg, label='Method 2: $\\langle f \\rangle (r)$', color='orange', linestyle='--')
plt.xlabel('Extension $r$')
plt.ylabel('Force')
plt.title('Scaled direct force vs average force')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 保存图像（请修改为你的路径）
output_path = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results/2-state.png'
plt.savefig(output_path, dpi=300)
print(f"图像已保存至: {output_path}")
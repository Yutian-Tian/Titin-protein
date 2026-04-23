import numpy as np
import matplotlib.pyplot as plt
from math import floor, pi

# 定义参数
L = 2000.0
a = 5.0
xi_values = [100.0, 200.0, 300.0, 500.0]

# 创建图形
plt.figure(figsize=(10, 6))

# 定义计算函数
def calculate_f(L, xi, n):
    Lc = L - n * xi
    alpha = xi / Lc
    beta = a / xi
    gamma = np.sqrt(alpha**2 + (beta**2 + 3 )*(alpha + 1))
    x_val = ((1 + alpha)* gamma + beta) / (3 + 3 * alpha + alpha**2)
    f_val = -pi**2 * x_val / Lc**2 + 4 * x_val / (pi * (1 - x_val**2))
    return f_val

# 对每个ξ值进行计算和绘图
for xi in xi_values:
    # 计算最大n值
    N_max = floor(L / xi) - 1
    n_values = list(range(0, N_max + 1))
    f_values = []
    
    # 计算每个n对应的f值
    for n in n_values:
        try:
            f_val = calculate_f(L, xi, n)
            f_values.append(f_val)
        except ZeroDivisionError:
            # 处理可能的除零错误
            break
    
    # 绘制曲线
    plt.plot(n_values[:len(f_values)], f_values, 'o-', label=f'ξ = {xi}')

# 添加图形标签和标题
plt.xlabel('$n_{f}$')
plt.ylabel('$f_{peak}$')
plt.title('$f_{peak}$ vs $n_{f}$ for different ξ values (L=2000, a=5)')
plt.legend()
plt.grid(True, alpha=0.3)

# 显示图形
plt.tight_layout()
plt.savefig('peak_value_n_f.png', dpi=300)
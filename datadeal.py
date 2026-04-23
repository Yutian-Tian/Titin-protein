import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 读取CSV文件（跳过第一行）
df1 = pd.read_csv('/home/tyt/project/Single-chain/iteration_results.csv', skiprows=2, header=None)


# 重命名列
df1.columns = ['r', 'f', 'Lc', 'pf']


# 可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(df1['r'].head(900), df1['f'].head(900), alpha=1)
plt.xlabel('$r$')
plt.ylabel('$f_a$')
plt.title('$f_a$ vs $r$')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(df1['r'], df1['Lc'], s=10, alpha=1)
plt.xlabel('$r$')
plt.ylabel('$Lc$')
plt.title('$L_c$ vs $r$')
plt.grid(True)



plt.savefig('/home/tyt/project/Single-chain/opt+R/fa_r.png', dpi=300)


print("处理完成！")
"""
系统：2个domain串联
方法：两个domain的独立优化
使用均匀扫描（网格搜索）
使用精确导数计算力
验证DeltaEi=1+0.5(xi_fi-5)，在[5,15]的范围内是否都满足domain按长度先后打卡
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm

# ============ 字体和样式设置 ============
# 指定字体路径
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

# 检查字体文件是否存在
if os.path.exists(font_path):
    # 将字体文件添加到matplotlib的字体管理器中
    fm.fontManager.addfont(font_path)
    # 获取字体的属性
    font_prop = fm.FontProperties(fname=font_path)
    # 将字体的名称设置为默认字体
    plt.rcParams['font.family'] = font_prop.get_name()
    print(f"使用字体: {font_prop.get_name()}")
else:
    print(f"警告: 字体文件不存在: {font_path}")
    print("将使用默认字体")

# 全局样式设置
plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'axes.labelsize': 35,
    'axes.titlesize': 35,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 25,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
})

# ============ 参数设置 ============
xi_f = np.linspace(5.0, 15.0, 11)  # 从5到15，共11个点
k = 2.0  # 展开系数
E0 = 1.0
alpha = 0.5

# 网格密度参数
r_grid = 200
n_grid = 200

# 创建保存文件夹
save_path = "/home/tyt/project/Single-chain/opt+R/Rand_xi/ordertest"
os.makedirs(save_path, exist_ok=True)

# ============ 定义函数 ============
def contour_length(xi_f1, xi_f2, n1, n2):
    """计算轮廓长度 L_c(n1, n2)"""
    xi_u1 = k * xi_f1
    xi_u2 = k * xi_f2
    L1 = xi_f1 + n1 * (xi_u1 - xi_f1)
    L2 = xi_f2 + n2 * (xi_u2 - xi_f2)
    return L1 + L2

def F_WLC(r, xi_f1, xi_f2, n1, n2):
    """WLC模型 Marko-Saggia"""
    Lc = contour_length(xi_f1, xi_f2, n1, n2)
    if r >= Lc:
        return np.inf  # 避免无效值
    numerator = r**2 * (3 * Lc - 2 * r)
    denominator = 4 * Lc * (Lc - r)
    return numerator / denominator

def U_energy(xi_f1, xi_f2, n1, n2):
    """能量项 U(n1, n2)"""
    DeltaE1 = E0 + alpha * (xi_f1 - 5.0)
    DeltaE2 = E0 + alpha * (xi_f2 - 5.0)
    U1 = DeltaE1
    U2 = DeltaE2
    term1 = DeltaE1 * n1 + DeltaE2 * n2
    term2 = U1 * np.cos(2 * np.pi * n1)
    term3 = U2 * np.cos(2 * np.pi * n2)
    return term1 - term2 - term3

def total_free_energy(r, xi_f1, xi_f2, n1, n2):
    """总自由能 F_c(r, n1, n2)"""
    return F_WLC(r, xi_f1, xi_f2, n1, n2) + U_energy(xi_f1, xi_f2, n1, n2)

def force_exact(r, xi_f1, xi_f2, n1, n2):
    """使用精确公式计算力"""
    Lc = contour_length(xi_f1, xi_f2, n1, n2)
    if r >= Lc:
        return np.inf
    
    # 精确公式：f(r, n1, n2) = (1/4) * { [1 - r/Lc]^{-2} - 1 + 4r/Lc }
    term = r / Lc
    force = 0.25 * ((1 - term) ** (-2) - 1 + 4 * term)
    return force

def Optimize(xi_f1, xi_f2):
    """优化函数，对于给定的xi_f1和xi_f2，计算最优解"""
    # 计算最大轮廓长度
    max_contour = contour_length(xi_f1, xi_f2, 1.0, 1.0)
    
    # r扫描范围
    r_min = 0.0  # 避免除零
    r_max = 0.95 * max_contour
    r_vals = np.linspace(r_min, r_max, r_grid)
    
    # n扫描范围
    n1_vals = np.linspace(0.0, 1.0, n_grid)
    n2_vals = np.linspace(0.0, 1.0, n_grid)
    
    # 初始化结果数组
    n1_opt = []
    n2_opt = []
    F_min = []
    forces = []
    
    for i in range(len(r_vals)):
        r = r_vals[i]
        
        min_F = np.inf
        min_n1 = 0
        min_n2 = 0
        
        # 网格搜索（均匀扫描）
        for n1 in n1_vals:
            for n2 in n2_vals:
                Lc = contour_length(xi_f1, xi_f2, n1, n2)
                if r < Lc:  # 只考虑r < Lc的情况
                    F = total_free_energy(r, xi_f1, xi_f2, n1, n2)
                    if F < min_F:
                        min_F = F
                        min_n1 = n1
                        min_n2 = n2
        
        # 保存结果
        n1_opt.append(min_n1)
        n2_opt.append(min_n2)
        F_min.append(min_F)
        
        # 使用精确公式计算力
        if min_F < np.inf:
            f = force_exact(r, xi_f1, xi_f2, min_n1, min_n2)
            if f == np.inf or np.isnan(f):
                f = 0
        else:
            f = 0
        forces.append(f)
    
    return r_vals, np.array(n1_opt), np.array(n2_opt), np.array(F_min), np.array(forces)

def save_results(xi_f1, xi_f2, r_vals, n1_opt, n2_opt, F_min, forces):
    """保存结果到文件"""
    filename = f"res_{xi_f1:.1f}_{xi_f2:.1f}.txt"
    filepath = os.path.join(save_path, filename)
    
    # 保存数据
    data = np.column_stack((r_vals, n1_opt, n2_opt, F_min, forces))
    header = f"r n1_opt n2_opt F_min force\nxi_f1={xi_f1} xi_f2={xi_f2}"
    np.savetxt(filepath, data, header=header, fmt='%.6f')
    
    print(f"结果已保存到: {filepath}")

def plot_results(xi_f1, xi_f2, r_vals, n1_opt, n2_opt, forces):
    """绘制2D散点图结果"""
    
    # 设置散点图的样式参数
    scatter_size = 35  # 散点大小
    alpha_value = 0.6  # 透明度
    edge_width = 0.5   # 边缘线宽
    
    # 图1: n1_opt和n2_opt随r的变化 - 散点图
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制n1的散点图 - 使用蓝色圆圈
    scatter1 = ax1.scatter(r_vals, n1_opt, s=scatter_size, alpha=alpha_value, 
                          c='blue', edgecolors='darkblue', linewidth=edge_width,
                          marker='o', label=f'$n_1$ ($\\xi_{{f1}}$={xi_f1})')
    
    # 绘制n2的散点图 - 使用红色三角形
    scatter2 = ax1.scatter(r_vals, n2_opt, s=scatter_size, alpha=alpha_value,
                          c='red', edgecolors='darkred', linewidth=edge_width,
                          marker='.', label=f'$n_2$ ($\\xi_{{f2}}$={xi_f2})')
    
    ax1.set_xlabel('End-to-end distance $r$', fontsize=14)
    ax1.set_ylabel('Unfolding parameters $n_i$', fontsize=14)
    ax1.set_title(f'Optimal $n_1$ and $n_2$ vs. $r$ ($\\xi_{{f1}}$={xi_f1}, $\\xi_{{f2}}$={xi_f2})', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0.0, r_vals[-1])
    ax1.set_ylim(-0.05, 1.05)
    
    # 设置刻度样式
    ax1.tick_params(axis='both', which='both', direction='in', width=1.0, length=4)
    
    # 添加背景网格线
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # 保存图像
    filename1 = f"n_{xi_f1:.1f}_{xi_f2:.1f}_scatter.png"
    filepath1 = os.path.join(save_path, filename1)
    plt.tight_layout()
    plt.savefig(filepath1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print(f"图像1已保存到: {filepath1}")
    
    # 图2: force随r的变化 - 散点图
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # 根据力的大小设置颜色映射
    if np.max(forces) > np.min(forces):
        norm_forces = (forces - np.min(forces)) / (np.max(forces) - np.min(forces))
        colors = plt.cm.viridis(norm_forces)  # 使用viridis颜色映射
    else:
        colors = 'green'  # 如果所有力都相同，使用单一颜色
    
    # 绘制力-延伸散点图
    scatter3 = ax2.scatter(r_vals, forces, s=scatter_size, alpha=alpha_value,
                          c=colors, edgecolors='black', linewidth=edge_width,
                          marker='o')
    
    ax2.set_xlabel('End-to-end distance $r$', fontsize=14)
    ax2.set_ylabel('Force $f$', fontsize=14)
    ax2.set_title(f'Force-extension curve ($\\xi_{{f1}}$={xi_f1}, $\\xi_{{f2}}$={xi_f2})', fontsize=16)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0.0, r_vals[-1])
    ax2.set_ylim(-1.0, 10.0)
    
    # 设置刻度样式
    ax2.tick_params(axis='both', which='both', direction='in', width=1.0, length=4)
    
    # 添加颜色条（只有当力的范围不为零时）
    if np.max(forces) > np.min(forces):
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                   norm=plt.Normalize(vmin=np.min(forces), 
                                                      vmax=np.max(forces)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2)
        cbar.set_label('Force intensity', rotation=270, labelpad=15, fontsize=12)
        cbar.ax.tick_params(labelsize=10)
    
    # 保存图像
    filename2 = f"force_{xi_f1:.1f}_{xi_f2:.1f}_scatter.png"
    filepath2 = os.path.join(save_path, filename2)
    plt.tight_layout()
    plt.savefig(filepath2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"图像2已保存到: {filepath2}")

# ============ 主程序 ============
def main():
    """主程序，迭代计算所有组合"""
    total_combinations = len(xi_f) * (len(xi_f) - 1) // 2  # 计算组合数（不包含相等的情况）
    
    print("=" * 80)
    print("双domain系统优化计算 - 2D散点图版本")
    print(f"ξ_f范围: [{xi_f[0]}, {xi_f[-1]}]")
    print(f"展开系数 k = {k}")
    print(f"网格参数: r_grid={r_grid}, n_grid={n_grid}")
    print(f"总组合数: {total_combinations}")
    print(f"保存路径: {save_path}")
    print("=" * 80)
    
    # 初始化结果汇总
    summary_results = []
    
    count = 0
    for i in range(len(xi_f)):
        xi_f1 = xi_f[i]
        for j in range(i + 1, len(xi_f)):
            xi_f2 = xi_f[j]
            count += 1
            
            print(f"\n{'='*40}")
            print(f"计算进度: {count}/{total_combinations}")
            print(f"当前参数: ξ_f1={xi_f1:.1f}, ξ_f2={xi_f2:.1f}")
            print(f"{'='*40}")
            
            # 计算DeltaEi
            DeltaE1 = E0 + alpha * (xi_f1 - 5.0)
            DeltaE2 = E0 + alpha * (xi_f2 - 5.0)
            print(f"能量参数: ΔE1={DeltaE1:.3f}, ΔE2={DeltaE2:.3f}")
            
            # 优化计算
            r_vals, n1_opt, n2_opt, F_min, forces = Optimize(xi_f1, xi_f2)
            
            # 保存结果
            save_results(xi_f1, xi_f2, r_vals, n1_opt, n2_opt, F_min, forces)
            
            # 绘制散点图
            plot_results(xi_f1, xi_f2, r_vals, n1_opt, n2_opt, forces)
            
            # 简单分析：计算平均值
            n1_mean = np.mean(n1_opt)
            n2_mean = np.mean(n2_opt)
            
            print(f"均值分析: n1均值={n1_mean:.4f}, n2均值={n2_mean:.4f}")
            
            # 保存到汇总结果
            summary_results.append({
                "xi_f1": xi_f1,
                "xi_f2": xi_f2,
                "DeltaE1": DeltaE1,
                "DeltaE2": DeltaE2,
                "n1_mean": n1_mean,
                "n2_mean": n2_mean
            })
    
    print("\n" + "=" * 80)
    print("所有计算完成！")
    print(f"结果保存在: {save_path}")
    print("=" * 80)
    
    # 保存汇总结果
    summary_file = os.path.join(save_path, "summary_results.txt")
    with open(summary_file, 'w') as f:
        f.write("双domain系统汇总\n")
        f.write("=" * 80 + "\n")
        f.write("序号 | ξ_f1 | ξ_f2 | ΔE1 | ΔE2 | n1均值 | n2均值\n")
        f.write("-" * 80 + "\n")
        
        for idx, result in enumerate(summary_results, 1):
            f.write(f"{idx:3d} | {result['xi_f1']:5.1f} | {result['xi_f2']:5.1f} | "
                   f"{result['DeltaE1']:5.2f} | {result['DeltaE2']:5.2f} | "
                   f"{result['n1_mean']:7.4f} | {result['n2_mean']:7.4f}\n")
    
    print(f"汇总结果已保存到: {summary_file}")
    
    # 创建参数文件记录实验设置
    param_file = os.path.join(save_path, "parameters.txt")
    with open(param_file, 'w') as f:
        f.write("双domain系统参数设置\n")
        f.write("=" * 60 + "\n")
        f.write(f"ξ_f范围: [{xi_f[0]}, {xi_f[-1]}], 共{len(xi_f)}个点\n")
        f.write(f"展开系数 k = {k}\n")
        f.write(f"能量公式: ΔEi = 1 + 0.5*(ξ_fi - 5)\n")
        f.write(f"网格参数: r_grid={r_grid}, n_grid={n_grid}\n")
        f.write(f"计算组合数: {total_combinations}\n")
        f.write(f"图像类型: 2D散点图\n")
        f.write(f"字体路径: {font_path}\n")
        f.write(f"计算时间: {np.datetime64('now')}\n")
        f.write("=" * 60 + "\n")
    
    print(f"参数文件已保存: {param_file}")

if __name__ == "__main__":
    main()
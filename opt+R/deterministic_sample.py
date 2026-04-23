import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib


matplotlib.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def gaussian_discretization(E_a, sigma, n_points=9):
    """
    基于中位点分位数法将高斯分布离散化为n个点
    
    参数:
    E_a: 高斯分布的均值
    sigma: 高斯分布的标准差
    n_points: 离散点的数量
    
    返回:
    points: 离散点的位置
    weights: 每个点的权重
    probabilities: 对应的分位概率
    """
    # 计算中位点分位数
    probabilities = [(k - 0.5) / n_points for k in range(1, n_points + 1)]
    
    # 计算对应的z值（标准正态分位数）
    z_values = norm.ppf(probabilities)
    
    # 转换到目标高斯分布
    points = E_a + sigma * np.array(z_values)
    
    # 均匀权重
    weights = np.ones(n_points) / n_points
    
    return points, weights, probabilities

def plot_discretization(E_a, sigma, points, weights, probabilities, save_path):
    """
    可视化高斯分布和离散方案并保存图像
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 第一个子图：概率密度函数和离散点
    x = np.linspace(E_a - 4*sigma, E_a + 4*sigma, 1000)
    y = norm.pdf(x, E_a, sigma)
    
    ax1.plot(x, y, 'b-', linewidth=2, label=f'Gaussian $N({E_a}, {sigma}²)$')
    ax1.fill_between(x, y, alpha=0.3, color='blue')
    
    # 标记离散点
    colors = plt.cm.viridis(np.linspace(0, 1, len(points)))
    for i, (point, weight, prob) in enumerate(zip(points, weights, probabilities)):
        ax1.axvline(x=point, color=colors[i], linestyle='--', alpha=0.7, 
                   label=f'$E_{i+1}$={point:.3f}, p={prob:.3f}')
        ax1.plot(point, norm.pdf(point, E_a, sigma), 'o', 
                color=colors[i], markersize=8, markeredgecolor='black')
    
    ax1.set_xlabel('$E$')
    ax1.set_ylabel('P(E)')
    ax1.set_title(f'Gaussian $N({E_a}, {sigma}²)$\'s 9-points results')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 第二个子图：累积分布函数和分位数
    y_cdf = norm.cdf(x, E_a, sigma)
    
    ax2.plot(x, y_cdf, 'r-', linewidth=2, label='CDF')
    
    # 标记分位数点
    for i, (point, prob) in enumerate(zip(points, probabilities)):
        ax2.axvline(x=point, color=colors[i], linestyle='--', alpha=0.7)
        ax2.plot(point, prob, 's', color=colors[i], markersize=8, 
                markeredgecolor='black', label=f'point {i+1}: p={prob:.3f}')
    
    ax2.set_xlabel('$E$')
    ax2.set_ylabel('$F(E)$')
    ax2.set_title('CDF and Results')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，释放内存
    print(f"图像已保存到: {save_path}")

def print_discretization_details(E_a, sigma, points, weights, probabilities):
    """
    打印离散化的详细信息
    """
    print("=" * 60)
    print(f"高斯分布 N({E_a}, {sigma}²) 的9点离散化方案")
    print("=" * 60)
    print(f"{'点号':<6} {'分位概率':<10} {'Z值':<10} {'能量值':<12} {'权重':<8}")
    print("-" * 60)
    
    z_values = (np.array(points) - E_a) / sigma
    
    for i, (prob, z, point, weight) in enumerate(zip(probabilities, z_values, points, weights)):
        print(f"{i+1:<6} {prob:<10.4f} {z:<10.4f} {point:<12.4f} {weight:<8.4f}")
    
    print("-" * 60)
    
    # 计算统计量
    discrete_mean = np.sum(points * weights)
    discrete_variance = np.sum(weights * (points - discrete_mean)**2)
    
    print(f"原始分布: 均值 = {E_a:.4f}, 方差 = {sigma**2:.4f}")
    print(f"离散分布: 均值 = {discrete_mean:.4f}, 方差 = {discrete_variance:.4f}")
    print(f"均值误差: {abs(discrete_mean - E_a):.6f}")
    print(f"方差误差: {abs(discrete_variance - sigma**2):.6f}")

def main():
    """
    主函数：处理真实的高斯分布
    """
    # 设置真实的高斯分布参数
    E_a = 10.0  # 均值
    sigma = 3.0 # 标准差
    
    # 文件保存路径
    save_path = "/home/tyt/project/Single-chain/opt+R/distr1.png"
    
    # 进行离散化
    points, weights, probabilities = gaussian_discretization(E_a, sigma)
    print(points)
    
    # 打印详细信息
    print_discretization_details(E_a, sigma, points, weights, probabilities)
    
    # 绘制并保存图像
    plot_discretization(E_a, sigma, points, weights, probabilities, save_path)

if __name__ == "__main__":
    # 使用方法1：使用主函数（默认参数）
    main()
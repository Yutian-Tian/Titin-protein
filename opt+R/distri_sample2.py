import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def truncated_gaussian_discretization(E_a, sigma, n_points=9):
    """
    离散化定义在 E >= 0 的截断高斯分布
    """
    # 计算截断分布的归一化常数
    # P(E>=0) = 1 - Φ(-E_a/sigma) = Φ(E_a/sigma)
    truncation_prob = norm.cdf(E_a/sigma)
    
    # 计算截断后的有效分位数
    # 原始分位数范围从 truncation_prob 的左侧开始
    probabilities = [truncation_prob * (k - 0.5) / n_points for k in range(1, n_points + 1)]
    
    # 计算对应的z值
    z_values = norm.ppf(probabilities)
    
    # 转换到目标高斯分布
    points = E_a + sigma * np.array(z_values)
    
    # 均匀权重（已经考虑了截断）
    weights = np.ones(n_points) / n_points
    
    return points, weights, probabilities

def truncated_gaussian_discretization_corrected(E_a, sigma, n_points=9):
    """
    更准确的截断高斯分布离散化方法
    """
    # 计算截断点对应的累积概率
    truncation_point = 0
    F0 = norm.cdf(truncation_point, E_a, sigma)  # P(E <= 0)
    
    # 在截断区间 [F0, 1] 内等分
    probabilities = [F0 + (1 - F0) * (k - 0.5) / n_points for k in range(1, n_points + 1)]
    
    # 计算对应的分位数
    points = norm.ppf(probabilities, E_a, sigma)
    
    # 均匀权重
    weights = np.ones(n_points) / n_points
    
    return points, weights, probabilities

def plot_truncated_discretization(E_a, sigma, points, weights, probabilities, save_path):
    """
    可视化截断高斯分布的离散化
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 第一个子图：概率密度函数
    x_min = max(0, E_a - 4*sigma)  # 从0开始
    x_max = E_a + 4*sigma
    x = np.linspace(x_min, x_max, 1000)
    
    # 截断高斯分布的PDF
    y_truncated = norm.pdf(x, E_a, sigma)
    y_truncated[x < 0] = 0  # E < 0 的部分设为0
    
    ax1.plot(x, y_truncated, 'b-', linewidth=2, label=f'Truncated Gaussian N({E_a}, {sigma}^2)')
    ax1.fill_between(x, y_truncated, alpha=0.3, color='blue')
    
    # 标记离散点
    colors = plt.cm.viridis(np.linspace(0, 1, len(points)))
    for i, (point, weight, prob) in enumerate(zip(points, weights, probabilities)):
        if point >= 0:  # 只显示非负的点
            ax1.axvline(x=point, color=colors[i], linestyle='--', alpha=0.7)
            ax1.plot(point, norm.pdf(point, E_a, sigma), 'o', 
                    color=colors[i], markersize=8, markeredgecolor='black',
                    label=f'Point {i+1}: E={point:.3f}')
    
    ax1.set_xlabel('Energy E')
    ax1.set_ylabel('Probability Density P(E)')
    ax1.set_title(f'9-Point Discretization of Truncated Gaussian (E >= 0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(x_min, x_max)
    
    # 第二个子图：累积分布函数
    y_cdf = norm.cdf(x, E_a, sigma)
    # 调整CDF：E < 0 的部分为0
    y_cdf_adj = np.where(x < 0, 0, (y_cdf - norm.cdf(0, E_a, sigma)) / (1 - norm.cdf(0, E_a, sigma)))
    
    ax2.plot(x, y_cdf_adj, 'r-', linewidth=2, label='Adjusted CDF (E >= 0)')
    
    # 标记分位数点
    for i, (point, prob) in enumerate(zip(points, probabilities)):
        if point >= 0:
            ax2.axvline(x=point, color=colors[i], linestyle='--', alpha=0.7)
            ax2.plot(point, (prob - norm.cdf(0, E_a, sigma)) / (1 - norm.cdf(0, E_a, sigma)), 's', 
                    color=colors[i], markersize=8, markeredgecolor='black',
                    label=f'Q{i+1}: p={prob:.3f}')
    
    ax2.set_xlabel('Energy E')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Adjusted Cumulative Distribution Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Image saved to: {save_path}")

def print_truncated_details(E_a, sigma, points, weights, probabilities):
    """
    打印截断分布的详细信息
    """
    print("=" * 70)
    print(f"9-Point Discretization of Truncated Gaussian N({E_a}, {sigma}^2), E >= 0")
    print("=" * 70)
    print(f"{'Point':<6} {'Quantile':<10} {'Z-value':<10} {'Energy':<12} {'Weight':<8} {'Valid':<8}")
    print("-" * 70)
    
    z_values = (np.array(points) - E_a) / sigma
    truncation_prob = norm.cdf(0, E_a, sigma)
    
    valid_points = 0
    for i, (prob, z, point, weight) in enumerate(zip(probabilities, z_values, points, weights)):
        valid = "Yes" if point >= 0 else "No"
        if point >= 0:
            valid_points += 1
        print(f"{i+1:<6} {prob:<10.4f} {z:<10.4f} {point:<12.4f} {weight:<8.4f} {valid:<8}")
    
    print("-" * 70)
    print(f"Truncation probability P(E < 0) = {truncation_prob:.4f}")
    print(f"Valid points (E >= 0): {valid_points}/{len(points)}")
    
    # 计算统计量（只考虑有效点）
    valid_mask = np.array(points) >= 0
    if np.any(valid_mask):
        valid_points = np.array(points)[valid_mask]
        valid_weights = np.array(weights)[valid_mask]
        valid_weights = valid_weights / np.sum(valid_weights)  # 重新归一化
        
        discrete_mean = np.sum(valid_points * valid_weights)
        discrete_variance = np.sum(valid_weights * (valid_points - discrete_mean)**2)
        
        # 截断高斯分布的理论矩
        alpha = -E_a/sigma
        Z = 1 - norm.cdf(alpha)  # 归一化常数
        theoretical_mean = E_a + sigma * norm.pdf(alpha) / Z
        theoretical_var = sigma**2 * (1 + alpha*norm.pdf(alpha)/Z - (norm.pdf(alpha)/Z)**2)
        
        print(f"Theoretical (truncated): mean = {theoretical_mean:.4f}, variance = {theoretical_var:.4f}")
        print(f"Discrete: mean = {discrete_mean:.4f}, variance = {discrete_variance:.4f}")
        print(f"Mean error: {abs(discrete_mean - theoretical_mean):.6f}")
        print(f"Variance error: {abs(discrete_variance - theoretical_var):.6f}")

def adaptive_truncated_discretization(E_a, sigma, n_points=9, max_iterations=10):
    """
    自适应方法：确保所有点都在 E >= 0 区域
    """
    # 计算截断概率
    F0 = norm.cdf(0, E_a, sigma)
    
    # 如果截断概率很小，可以直接使用标准方法
    if F0 < 0.01:
        return truncated_gaussian_discretization_corrected(E_a, sigma, n_points)
    
    # 否则使用自适应方法
    probabilities = []
    points = []
    
    # 从截断点开始等间隔
    prob_range = 1 - F0
    for k in range(1, n_points + 1):
        prob = F0 + prob_range * (k - 0.5) / n_points
        point = norm.ppf(prob, E_a, sigma)
        
        # 如果点仍然为负，调整概率范围
        if point < 0 and k == 1:
            # 找到第一个正点对应的概率
            prob_first_positive = norm.cdf(0, E_a, sigma) + 1e-10
            prob_range = 1 - prob_first_positive
            prob = prob_first_positive + prob_range * (k - 0.5) / n_points
            point = norm.ppf(prob, E_a, sigma)
        
        probabilities.append(prob)
        points.append(point)
    
    weights = np.ones(n_points) / n_points
    
    return np.array(points), weights, probabilities

def main():
    """
    主函数：演示截断高斯分布的离散化
    """
    # 测试不同的参数
    test_cases = [
        (10.0, 8.0),   # 均值较大，截断影响小
    #    (1.0, 1.0),   # 均值等于标准差
    #    (0.5, 1.0),   # 均值较小，截断影响显著
    ]
    
    for i, (E_a, sigma) in enumerate(test_cases):
        print(f"\n{'#'*60}")
        print(f"Test Case {i+1}: E_a = {E_a}, sigma = {sigma}")
        print(f"{'#'*60}")
        
        # 文件保存路径
        save_path = f"/home/tyt/project/Single-chain/opt+R/truncated_distr_{i+2}.png"
        
        # 使用自适应方法进行离散化
        points, weights, probabilities = adaptive_truncated_discretization(E_a, sigma)
        print(points)
        
        # 打印详细信息
        print_truncated_details(E_a, sigma, points, weights, probabilities)
        
        # 绘制并保存图像
        plot_truncated_discretization(E_a, sigma, points, weights, probabilities, save_path)

if __name__ == "__main__":
    main()
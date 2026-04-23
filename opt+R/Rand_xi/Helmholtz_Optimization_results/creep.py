"""
求解无量纲粘弹性积分方程
     σ̄ = e^{-τ}(λ - 1/λ²) + ∫₀^τ e^{-(τ-τ')} [ λ/λ'(τ')² - λ(τ')/λ² ] dτ'
得到恒定应力下的蠕变曲线 λ(τ)
方法：左矩形递推 + 牛顿迭代
"""

import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# 核心求解函数（左矩形递推法，O(N) 高效）
# ==================================================
def solve_creep_dimensionless(sigma_bar, tau_max, N, tol=1e-10, max_iter=50):
    """
    使用左矩形递推法求解无量纲蠕变方程。

    参数：
        sigma_bar : 无量纲应力 σ/G0（常数）
        tau_max   : 最大无量纲时间
        N         : 时间步数
        tol       : 牛顿迭代收敛容差
        max_iter  : 最大迭代次数

    返回：
        tau : 无量纲时间数组 (长度 N+1)
        lam : 伸长率数组 (长度 N+1)
    """
    tau = np.linspace(0, tau_max, N+1)
    dtau = tau[1] - tau[0]
    lam = np.zeros(N+1)

    # ---------- 初始时刻 τ = 0 ----------
    def F0(lam_val):
        return (lam_val - 1.0/lam_val**2) - sigma_bar
    def dF0(lam_val):
        return 1.0 + 2.0/lam_val**3

    # 牛顿法求 λ₀
    lam_guess = 1.0
    while True:
        f = F0(lam_guess)
        df = dF0(lam_guess)
        lam_new = lam_guess - f/df
        if abs(lam_new - lam_guess) < tol:
            break
        lam_guess = lam_new
    lam[0] = lam_new

    # 积分递推量 A, B 初值
    A = 0.0   # A_n = ∫ e^{-(τ_n-τ')} / λ²(τ') dτ'
    B = 0.0   # B_n = ∫ e^{-(τ_n-τ')} λ(τ') dτ'
    exp_dtau = np.exp(-dtau)

    # ---------- 时间步进 ----------
    for n in range(1, N+1):
        # 左矩形递推更新 A, B
        A = exp_dtau * A + dtau * exp_dtau / (lam[n-1]**2)
        B = exp_dtau * B + dtau * exp_dtau * lam[n-1]

        exp_tau_n = np.exp(-tau[n])

        # 牛顿法求解 λ_n
        lam_k = lam[n-1]
        for _ in range(max_iter):
            lam2 = lam_k**2
            lam3 = lam_k**3

            # 方程 F(λ) = 0
            F_val = (exp_tau_n * (lam_k - 1.0/lam2) +
                     lam_k * A - B / lam2 - sigma_bar)

            # 导数 dF/dλ
            dF_val = (exp_tau_n * (1.0 + 2.0/lam3) +
                      A + 2.0 * B / lam3)

            lam_new = lam_k - F_val / dF_val
            if abs(lam_new - lam_k) < tol:
                lam[n] = lam_new
                break
            lam_k = lam_new
        else:
            print(f"  警告：τ = {tau[n]:.4f} 处牛顿法未收敛，残差 = {F_val:.3e}")
            lam[n] = lam_new   # 即使未收敛也继续

    return tau, lam


# ==================================================
# 可选：直接矩形求和版本（用于验证，O(N²)）
# ==================================================
def solve_by_direct_sum(sigma_bar, tau_max, N, tol=1e-10, max_iter=50):
    """
    使用直接左矩形求和（无递推）求解，更直观但较慢。
    """
    tau = np.linspace(0, tau_max, N+1)
    dtau = tau[1] - tau[0]
    lam = np.zeros(N+1)

    # 初值求解
    def F0(lam_val): return (lam_val - 1.0/lam_val**2) - sigma_bar
    def dF0(lam_val): return 1.0 + 2.0/lam_val**3
    lam_guess = 1.0
    while True:
        f = F0(lam_guess)
        df = dF0(lam_guess)
        lam_new = lam_guess - f/df
        if abs(lam_new - lam_guess) < tol:
            break
        lam_guess = lam_new
    lam[0] = lam_new

    for n in range(1, N+1):
        tau_n = tau[n]
        exp_tau_n = np.exp(-tau_n)

        # 直接求和计算系数 C_n, D_n
        C_sum = 0.0
        D_sum = 0.0
        for j in range(n):
            exp_diff = np.exp(-(tau_n - tau[j]))
            C_sum += exp_diff / (lam[j]**2)
            D_sum += exp_diff * lam[j]

        C_n = exp_tau_n + dtau * C_sum
        D_n = exp_tau_n + dtau * D_sum

        lam_k = lam[n-1]
        for _ in range(max_iter):
            lam2 = lam_k**2
            lam3 = lam_k**3
            F_val = lam_k * C_n - D_n / lam2 - sigma_bar
            dF_val = C_n + 2.0 * D_n / lam3
            lam_new = lam_k - F_val / dF_val
            if abs(lam_new - lam_k) < tol:
                lam[n] = lam_new
                break
            lam_k = lam_new
        else:
            print(f"  警告：τ = {tau_n:.4f} 处未收敛")
            lam[n] = lam_new

    return tau, lam


# ==================================================
# 主程序：批量计算、绘图、保存数据
# ==================================================
if __name__ == "__main__":
    # ------------------- 参数设置 -------------------
    sigma_bars = [0.1, 0.3, 0.5, 0.8, 1.2]   # 多个无量纲应力值
    tau_max = 10.0                           # 最大无量纲时间
    N = 800                                  # 时间步数
    save_data = True                         # 是否保存结果到文件

    # 颜色循环
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sigma_bars)))

    # 创建图形：左为线性坐标，右为半对数坐标
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ------------------- 对每个应力值计算 -------------------
    for i, sigma in enumerate(sigma_bars):
        print(f"正在计算 σ̄ = {sigma} ...", end=" ")
        tau, lam = solve_creep_dimensionless(sigma, tau_max, N)
        print("完成")

        # 绘制曲线
        ax1.plot(tau, lam, color=colors[i], linewidth=2, label=f'σ̄ = {sigma}')
        ax2.semilogy(tau, lam, color=colors[i], linewidth=2, label=f'σ̄ = {sigma}')

        # 可选：保存数据
        if save_data:
            np.savetxt(f"creep_sigma_{sigma:.2f}.txt",
                       np.column_stack((tau, lam)),
                       header="tau\tlambda", fmt="%.6f")

    # ------------------- 图表装饰 -------------------
    ax1.set_xlabel('无量纲时间 τ = βt', fontsize=12)
    ax1.set_ylabel('伸长率 λ', fontsize=12)
    ax1.set_title('恒定应力蠕变响应（线性坐标）', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.set_xlabel('无量纲时间 τ = βt', fontsize=12)
    ax2.set_ylabel('伸长率 λ', fontsize=12)
    ax2.set_title('恒定应力蠕变响应（半对数坐标）', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("creep_curves.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ------------------- 输出初始伸长率与末期伸长率 -------------------
    print("\n应力值\t初始伸长率 λ(0⁺)\t最终伸长率 λ(τ_max)")
    for sigma in sigma_bars:
        tau, lam = solve_creep_dimensionless(sigma, tau_max, N)
        print(f"{sigma:.2f}\t{lam[0]:.6f}\t\t{lam[-1]:.6f}")

    print("\n所有计算完成，图表已保存为 creep_curves.png")
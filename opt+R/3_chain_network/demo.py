# 将平均的力-拉伸曲线用于3-chain 网络模型，计算本构方程

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

class ConstitutiveModel:
    """本构模型计算器：从f-r关系计算σ-λ关系"""
    
    def __init__(self, config):
        """
        初始化本构模型
        
        参数:
            config: 配置字典，包含模型参数
        """
        # 模型参数
        self.L = 100.0 # 初始链长
        self.rho = config.get('rho', 1.0)      # 密度
        self.R0 = config.get('R0', 1.0)        # 初始端距
        self.kB = config.get('kB', 1.0)        # 玻尔兹曼常数
        self.T = config.get('T', 300.0)        # 温度
        
        # 文件路径
        self.f_r_file = config.get('f_r_file', 'force_data.csv')
        self.output_dir = config.get('output_dir', './results')
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 读取数据
        self.load_force_data()
        
        # 创建插值函数
        self.create_interpolator()
        
    def load_force_data(self):
        """从文件加载f-r数据"""
        print(f"从文件加载f-r数据: {self.f_r_file}")
        
        try:
            # 读取CSV文件
            data = pd.read_csv(self.f_r_file)
            
            # 检查列名
            if 'r' in data.columns and 'f' in data.columns:
                self.r_data = data['r'].values
                self.f_data = data['f'].values
            elif len(data.columns) >= 2:
                # 假设前两列是r和f
                self.r_data = data.iloc[:, 0].values
                self.f_data = data.iloc[:, 1].values
            else:
                raise ValueError("CSV文件格式不正确")
                
        except Exception as e:
            print(f"加载文件失败: {e}")
            raise
        
        # 确保数据是排序的
        sort_idx = np.argsort(self.r_data)
        self.r_data = self.r_data[sort_idx]
        self.f_data = self.f_data[sort_idx]
        
        print(f"  加载了 {len(self.r_data)} 个数据点")
        print(f"  r范围: {self.r_data[0]:.4f} 到 {self.r_data[-1]:.4f}")
        print(f"  f范围: {self.f_data.min():.4f} 到 {self.f_data.max():.4f}")
    
    def create_interpolator(self):
        """创建分段线性插值函数"""
        print("创建分段线性插值函数...")
        
        # 使用scipy的interp1d进行分段线性插值
        self.f_interp = interp1d(self.r_data, self.f_data, 
                                kind='linear', 
                                bounds_error=False,
                                fill_value=(self.f_data[0], self.f_data[-1]))
        
        # 设置外推方法
        self.r_min = self.r_data[0]
        self.r_max = self.r_data[-1]
        
        # 计算边缘的斜率用于外推
        if len(self.r_data) >= 2:
            self.slope_left = (self.f_data[1] - self.f_data[0]) / (self.r_data[1] - self.r_data[0])
            self.slope_right = (self.f_data[-1] - self.f_data[-2]) / (self.r_data[-1] - self.r_data[-2])
        else:
            self.slope_left = 0
            self.slope_right = 0
        
        print("  插值函数创建完成")
    
    def f(self, r):
        """
        计算力f(r)，支持内插和外推
        
        参数:
            r: 端距值或数组
            
        返回:
            力值或数组
        """
        r = np.asarray(r)
        result = np.zeros_like(r)
        
        # 内插区域
        mask_interp = (r >= self.r_min) & (r <= self.r_max)
        if np.any(mask_interp):
            result[mask_interp] = self.f_interp(r[mask_interp])
        
        # 左外推 (r < r_min)
        mask_left = r < self.r_min
        if np.any(mask_left):
            result[mask_left] = self.f_data[0] + self.slope_left * (r[mask_left] - self.r_min)
        
        # 右外推 (r > r_max)
        mask_right = r > self.r_max
        if np.any(mask_right):
            result[mask_right] = self.f_data[-1] + self.slope_right * (r[mask_right] - self.r_max)
        
        # 确保力非负
        result = np.maximum(result, 0)
        
        return result
    
    def sigma(self, lambda_val):
        """
        计算应力σ(λ)
        
        根据公式: σ(λ) = ρ R₀ k_B T [f(λR₀) - λ^{-3/2} f(λ^{-1/2} R₀)]
        
        参数:
            lambda_val: 拉伸比或数组
            
        返回:
            应力值或数组
        """
        lambda_val = np.asarray(lambda_val)
        
        # 确保λ > 0
        lambda_val = np.maximum(lambda_val, 1e-6)
        
        # 计算两个项
        term1 = self.f(lambda_val * self.R0)  # f(λR₀)
        term2 = self.f(lambda_val**(-0.5) * self.R0)  # f(λ^{-1/2} R₀)
        
        # 计算应力
        sigma_val = self.rho * self.R0 * self.kB * self.T * (
            term1 - lambda_val**(-1.5) * term2
        )
        
        # 处理可能出现的无穷大或NaN值
        sigma_val = np.nan_to_num(sigma_val, nan=0.0, posinf=0.0, neginf=0.0)
        
        return sigma_val
    
    def WLCsigma(self, lambda_val):
        """
        计算应力σ(λ)
        
        根据公式: σ(λ) = ρ R₀ k_B T [f(λR₀) - λ^{-3/2} f(λ^{-1/2} R₀)]
        
        参数:
            lambda_val: 拉伸比或数组
            
        返回:
            应力值或数组
        """
        lambda_val = np.asarray(lambda_val)
        
        # 确保λ > 0
        lambda_val = np.maximum(lambda_val, 1e-6)
        
        # 计算两个项
        term1 = self.WLCf(lambda_val * self.R0)  # f(λR₀)
        term2 = self.WLCf(lambda_val**(-0.5) * self.R0)  # f(λ^{-1/2} R₀)
        
        # 计算应力
        WLCsigma_val = self.rho * self.R0 * self.kB * self.T * (
            term1 - lambda_val**(-1.5) * term2
        )
        
        # 处理可能出现的无穷大或NaN值
        WLCsigma_val = np.nan_to_num(WLCsigma_val, nan=0.0, posinf=0.0, neginf=0.0)
        
        return WLCsigma_val
    
    def WLCf(self, r: float) -> float:
            """计算力: ∂F/∂r"""
            Lc = self.L
            if Lc <= 0:
                return np.nan
                
            x_val = r / Lc
            
            # 避免除零
            denominator = 1 - x_val
            if denominator.any() < 1e-10:
                return np.nan
            
            d_term = 0.25*((1 - x_val)**(-2) - 1 + 4*x_val)
            
            return d_term
    
    def calculate_constitutive_curve(self, lambda_min=1.0, lambda_max=3.0, n_points=200):
        """
        计算本构曲线
        
        参数:
            lambda_min: 最小拉伸比
            lambda_max: 最大拉伸比
            n_points: 采样点数
            
        返回:
            (lambda_vals, sigma_vals) 元组
        """
        print(f"\n计算本构曲线...")
        print(f"  拉伸比范围: {lambda_min} 到 {lambda_max}")
        print(f"  采样点数: {n_points}")
        
        # 生成拉伸比数组
        self.lambda_vals = np.linspace(lambda_min, lambda_max, n_points)
        
        # 计算应力
        self.sigma_vals = self.sigma(self.lambda_vals)
        self.WLCsigma_vals = self.WLCsigma(self.lambda_vals[0:150])
        
        # 移除无效值
        valid_mask = np.isfinite(self.sigma_vals)
        self.lambda_vals = self.lambda_vals[valid_mask]
        self.sigma_vals = self.sigma_vals[valid_mask]
        
        print(f"  计算完成，有效点数: {len(self.sigma_vals)}")
        
        return self.lambda_vals, self.sigma_vals
    
    def save_results(self):
        """保存计算结果"""
        print(f"\n保存结果到目录: {self.output_dir}")
        
        # 保存本构曲线
        constitutive_file = os.path.join(self.output_dir, "constitutive_curve.csv")
        np.savetxt(constitutive_file, 
                  np.column_stack([self.lambda_vals, self.sigma_vals]),
                  delimiter=',', header='lambda,sigma', fmt='%.6e')
        print(f"✅ 本构曲线已保存: {constitutive_file}")
        
        return constitutive_file
    
    def plot_constitutive_curve(self, save_fig=True):
        """绘制本构曲线"""
        print("\n绘制本构曲线...")
        
        # 创建图形
        plt.figure(figsize=(10, 6))
        
        # 绘制本构曲线
        plt.plot(self.lambda_vals, self.sigma_vals, 'b-', linewidth=2, label = 'Our model')
        plt.plot(self.lambda_vals[0:150], self.WLCsigma_vals, 'r-', linewidth=2, label = '3-chain WLC')
        plt.xlabel('λ', fontsize=14)
        plt.ylabel('stress $σ(λ)$', fontsize=14)
        plt.title('$\sigma$ vs $\lambda$', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
    
        # 标记最大应力点
        if len(self.sigma_vals) > 0:
            max_sigma_idx = np.argmax(self.sigma_vals)
            max_lambda = self.lambda_vals[max_sigma_idx]
            max_sigma = self.sigma_vals[max_sigma_idx]
            
            plt.plot(max_lambda, max_sigma, 'ro', markersize=8)
            plt.annotate(f'λ={max_lambda:.2f}\nσ={max_sigma:.2e}', 
                        xy=(max_lambda, max_sigma),
                        xytext=(max_lambda+0.1, max_sigma*0.9),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=10, color='red')
        
        plt.tight_layout()
        
        # 保存图形
        if save_fig:
            curve_file = os.path.join(self.output_dir, "constitutive_curve.png")
            plt.savefig(curve_file, dpi=300, bbox_inches='tight')
            print(f"✅ 本构曲线图已保存: {curve_file}")
        
        plt.show()
    
    def run(self, lambda_min=1.0, lambda_max=3.0, n_points=200, plot=True):
        """
        运行完整的计算流程
        
        参数:
            lambda_min: 最小拉伸比
            lambda_max: 最大拉伸比
            n_points: 采样点数
            plot: 是否绘制图形
        """
        print("=" * 60)
        print("开始计算本构关系")
        print("=" * 60)
        
        # 1. 计算本构曲线
        self.calculate_constitutive_curve(lambda_min, lambda_max, n_points)
        
        # 2. 保存结果
        saved_file = self.save_results()
        
        # 3. 绘制图形
        if plot:
            self.plot_constitutive_curve(save_fig=True)
        
        # 4. 打印统计信息
        self.print_statistics()
        
        print("\n" + "=" * 60)
        print("计算完成!")
        print("=" * 60)
        
        return self.lambda_vals, self.sigma_vals, saved_file
    
    def print_statistics(self):
        """打印统计信息"""
        if not hasattr(self, 'sigma_vals'):
            return
        
        print("\n=== 计算结果统计 ===")
        print(f"拉伸比范围: {self.lambda_vals[0]:.4f} 到 {self.lambda_vals[-1]:.4f}")
        print(f"应力范围: {self.sigma_vals.min():.4e} 到 {self.sigma_vals.max():.4e}")
        
        # 计算最大应力点
        max_sigma_idx = np.argmax(self.sigma_vals)
        max_lambda = self.lambda_vals[max_sigma_idx]
        max_sigma = self.sigma_vals[max_sigma_idx]
        print(f"最大应力: σ_max = {max_sigma:.4e} (λ = {max_lambda:.4f})")


def main():
    """主函数"""
    
    # 配置参数
    config = {
        # 模型参数
        'rho': 1.0,          # 密度
        'R0': 50.0,         # 初始端距
        'kB': 1.0,           # 玻尔兹曼常数
        'T': 1.0,            # 温度
        
        # 文件路径
        'f_r_file': '/home/tyt/project/Single-chain/opt+R/3_chain_network/average_curve_f_filtered_10.0.csv',
        'output_dir': '/home/tyt/project/Single-chain/opt+R/3_chain_network/constitutive_results'
    }
    
    # 创建模型实例
    model = ConstitutiveModel(config)
    
    # 运行计算
    lambda_vals, sigma_vals, saved_file = model.run(
        lambda_min=1.0,      # 最小拉伸比
        lambda_max=5.0,      # 最大拉伸比
        n_points=1000,       # 采样点数
        plot=True           # 绘制图形
    )
    
    return model, lambda_vals, sigma_vals, saved_file


if __name__ == "__main__":
    # 运行主程序
    model, lambda_vals, sigma_vals, saved_file = main()
    
    # 显示结果文件路径
    print(f"\n本构曲线数据已保存到: {saved_file}")
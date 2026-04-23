import numpy as np
import pandas as pd
import os
from typing import List, Dict, Tuple


class GaussianOptimizer:
    """高斯分布参数优化器"""
    
    def __init__(self, config: Dict):
        """
        初始化优化器
        
        参数:
            config: 配置字典，包含所有参数
        """
        # 基本参数
        self.a = config.get('a', 0.0)
        self.U0 = config.get('U0', 10.0)
        self.xi = config.get('xi', 30.0)
        
        # L的高斯分布参数
        self.L_mu = config.get('L_mu', 400.0)
        self.L_sigma = config.get('L_sigma', 50.0)
        
        # E的高斯分布参数 (10个能量值)
        self.E_mu = config.get('E_mu', 10.0)
        self.E_sigma = config.get('E_sigma', 1000.0)
        
        # 运行参数
        self.num_samples = config.get('num_samples', 100)
        self.r_points = config.get('r_points', 1000)
        self.scan_points = config.get('scan_points', 1000)
        
        # 输出目录
        self.output_dir = config.get('output_dir', './results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 存储原始采样数据
        self.samples_file = os.path.join(self.output_dir, "all_samples.txt")
        self.filtered_file = os.path.join(self.output_dir, "filtered_samples.txt")

         # 在初始化时清理目录
        self.clear_previous_results = config.get('clear_previous', True)
        if self.clear_previous_results:
            self.clear_output_files()

    def clear_output_files(self):
        """清空输出文件（保留目录结构）"""
        print("清理之前的输出文件...")
        
        # 要清理的文件列表
        files_to_clear = [
            self.samples_file,
            self.filtered_file,
            os.path.join(self.output_dir, "L_values.txt"),
            os.path.join(self.output_dir, "E_values.csv"),
            os.path.join(self.output_dir, "r_values.csv"),
            os.path.join(self.output_dir, "f_values.csv"),
            os.path.join(self.output_dir, "n_values.csv"),
            os.path.join(self.output_dir, "config.txt")
        ]
        
        for filepath in files_to_clear:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"  已删除: {os.path.basename(filepath)}")
                except Exception as e:
                    print(f"  警告: 无法删除 {filepath}: {e}")

    def generate_gaussian_samples(self) -> List[Dict]:
        """
        生成L和E的高斯分布样本
        
        返回:
            样本字典列表，每个包含L和E_array
        """
        print(f"生成 {self.num_samples} 个样本...")
        
        all_samples = []
        valid_samples = []
        
        # 生成L的样本
        L_samples = np.random.normal(self.L_mu, self.L_sigma, self.num_samples)
        
        # 生成E的样本 (每个样本有10个能量值)
        E_samples = np.random.normal(self.E_mu, self.E_sigma, (self.num_samples, 10))
        
        # 组合样本并筛选
        for i in range(self.num_samples):
            L = L_samples[i]
            E_array = E_samples[i]
            
            # 记录所有样本
            all_samples.append({
                'L': L,
                'E': E_array,
                'index': i
            })
            
            # 筛选条件：L > 10*xi + 10 且所有E > 0
            if L > (10 * self.xi + 10.0) and np.all(E_array > 0):
                valid_samples.append({
                    'L': L,
                    'E': E_array,
                    'index': i
                })
                
                # 保存到文件
                self._save_sample_to_file(L, E_array, self.samples_file)
        
        print(f"总样本数: {self.num_samples}")
        print(f"有效样本数: {len(valid_samples)}")
        
        # 保存有效样本到单独文件
        self._save_filtered_samples(valid_samples)
        
        return valid_samples
    
    def _save_sample_to_file(self, L: float, E_array: np.ndarray, filepath: str):
        """保存单个样本到文件"""
        E_str = ' '.join([f'{x:.6f}' for x in E_array])
        with open(filepath, 'a') as f:
            f.write(f"{L:.6f} {E_str}\n")
    
    def _save_filtered_samples(self, samples: List[Dict]):
        """保存筛选后的样本"""
        with open(self.filtered_file, 'w') as f:
            for sample in samples:
                L = sample['L']
                E_array = sample['E']
                E_str = ' '.join([f'{x:.6f}' for x in E_array])
                f.write(f"{L:.6f} {E_str}\n")
    
    def optimization_function(self, L: float, E_array: np.ndarray) -> Dict:
        """
        优化函数，计算给定L和E的自由能
        
        参数:
            L: 链长
            E_array: 能量数组 (10个值)
            
        返回:
            包含优化结果的字典
        """
        # 准备数据
        data1 = np.array(E_array)
        data1 = np.insert(data1, 0, 0.0)  # 在开头插入0
        data2 = np.linspace(0, 10, 11)
        data3 = np.cumsum(data1)  # 累积和
        
        # n的取值范围
        n_min = 0.0
        n_max = float(len(data2) - 1)
        
        # 定义F函数
        def F(n: float, r: float) -> float:
            """自由能函数"""
            Lc = L - (n_max - n) * self.xi
            if Lc <= 0:
                return np.inf
                
            c_val = 1 / (2 * Lc)
            x_val = (r - (n_max - n) * self.a) / Lc
            
            # 避免除零
            denominator = 1 - x_val**2
            if abs(denominator) < 1e-10:
                return np.inf
            
            term1 = np.pi**2 * c_val * denominator
            term2 = 1 / (np.pi * c_val * denominator + 1e-16)
            term3 = np.interp(n, data2, data3)
            term4 = -self.U0 * np.cos(2 * np.pi * n)
            
            return term1 + term2 + term3 + term4
        
        # 计算F对r的精确偏导数（力）
        def exact_partial_F_r(n: float, r: float) -> float:
            """计算力: ∂F/∂r"""
            Lc = L - (n_max - n) * self.xi
            if Lc <= 0:
                return np.nan
                
            x_val = (r - (n_max - n) * self.a) / Lc
            
            # 避免除零
            denominator = 1 - x_val**2
            if abs(denominator) < 1e-10:
                return np.nan
            
            d_term1_dr = -np.pi**2 * x_val / (Lc**2)
            d_term2_dr = 4 * x_val / (np.pi * (denominator**2))
            
            return d_term1_dr + d_term2_dr
        
        # 生成r的取值范围
        r_max = L - n_min * self.xi - 5.0
        if r_max <= n_max * self.a:
            return {
                'L': L,
                'r_values': np.array([]),
                'optimal_ns': np.array([]),
                'min_F_values': np.array([]),
                'force': np.array([]),
                'E_array': E_array
            }
        
        r_values = np.linspace(n_max * self.a, r_max, self.r_points)
        
        # 存储结果
        optimal_ns = []
        min_F_values = []
        partial_F_r_values = []
        
        # 对每个r值进行优化
        for r in r_values:
            # 计算n的实际取值范围
            n_lower_bound = max(n_min, n_max - (L - r) / self.xi)
            
            if n_lower_bound >= n_max:
                optimal_ns.append(np.nan)
                min_F_values.append(np.nan)
                partial_F_r_values.append(np.nan)
                continue
            
            # 在n的取值范围内均匀采样
            n_samples = np.linspace(n_lower_bound, n_max, self.scan_points)
            F_samples = np.full_like(n_samples, np.inf)
            
            # 计算每个采样点的F值
            for j, n in enumerate(n_samples):
                try:
                    F_val = F(n, r)
                    F_samples[j] = F_val
                except:
                    continue
            
            # 找到最小F值对应的n
            valid_indices = np.isfinite(F_samples)
            if not np.any(valid_indices):
                optimal_ns.append(np.nan)
                min_F_values.append(np.nan)
                partial_F_r_values.append(np.nan)
                continue
            
            min_index = np.nanargmin(F_samples)
            min_F = F_samples[min_index]
            optimal_n = n_samples[min_index]
            
            optimal_ns.append(optimal_n)
            min_F_values.append(min_F)
            
            # 计算力
            try:
                force = exact_partial_F_r(optimal_n, r)
                partial_F_r_values.append(force)
            except:
                partial_F_r_values.append(np.nan)
        
        # 转换为numpy数组
        optimal_ns = np.array(optimal_ns)
        min_F_values = np.array(min_F_values)
        partial_F_r_values = np.array(partial_F_r_values)
        
        # 创建结果字典
        result = {
            'L': L,
            'r_values': r_values,
            'optimal_ns': optimal_ns,
            'min_F_values': min_F_values,
            'force': partial_F_r_values,
            'E_array': E_array
        }
        
        return result
    
    def save_results_column_format(self, results_list: List[Dict]):
        """
        按列保存结果到文件
        
        参数:
            results_list: 结果字典列表
        """
        print(f"\n保存 {len(results_list)} 个样本的结果...")
        
        # 确保目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. 保存L值到txt文件
        L_values = [result['L'] for result in results_list]
        L_filepath = os.path.join(self.output_dir, "L_values.txt")
        np.savetxt(L_filepath, L_values, fmt='%.6f')
        print(f"✅ L值已保存到 {L_filepath}")
        
        # 2. 保存E值到CSV文件 (每个样本10个能量值)
        E_arrays = [result['E_array'] for result in results_list]
        E_df = pd.DataFrame(E_arrays)
        E_filepath = os.path.join(self.output_dir, "E_values.csv")
        E_df.to_csv(E_filepath, index=False, header=[f"E_{i}" for i in range(10)])
        print(f"✅ E值已保存到 {E_filepath}")
        
        # 3. 准备其他数据
        all_r_data = []
        all_f_data = []
        all_n_data = []
        
        # 找到最大数据长度
        max_length = max(len(result['r_values']) for result in results_list)
        
        for result in results_list:
            r_values = result['r_values']
            f_values = result['force']
            n_values = result['optimal_ns']
            
            # 填充到最大长度（用NaN填充）
            if len(r_values) < max_length:
                r_padded = np.full(max_length, np.nan)
                r_padded[:len(r_values)] = r_values
                
                f_padded = np.full(max_length, np.nan)
                f_padded[:len(f_values)] = f_values
                
                n_padded = np.full(max_length, np.nan)
                n_padded[:len(n_values)] = n_values
            else:
                r_padded = r_values
                f_padded = f_values
                n_padded = n_values
            
            all_r_data.append(r_padded)
            all_f_data.append(f_padded)
            all_n_data.append(n_padded)
        
        # 转置，使每列对应一个样本
        r_matrix = np.column_stack(all_r_data)
        f_matrix = np.column_stack(all_f_data)
        n_matrix = np.column_stack(all_n_data)
        
        # 4. 保存r值到CSV
        r_filepath = os.path.join(self.output_dir, "r_values.csv")
        r_columns = [f"sample_{i+1}" for i in range(len(results_list))]
        r_df = pd.DataFrame(r_matrix, columns=r_columns)
        r_df.to_csv(r_filepath, index=False)
        print(f"✅ r值已保存到 {r_filepath}")
        
        # 5. 保存f值到CSV
        f_filepath = os.path.join(self.output_dir, "f_values.csv")
        f_columns = [f"sample_{i+1}" for i in range(len(results_list))]
        f_df = pd.DataFrame(f_matrix, columns=f_columns)
        f_df.to_csv(f_filepath, index=False)
        print(f"✅ f值已保存到 {f_filepath}")
        
        # 6. 保存n值到CSV
        n_filepath = os.path.join(self.output_dir, "n_values.csv")
        n_columns = [f"sample_{i+1}" for i in range(len(results_list))]
        n_df = pd.DataFrame(n_matrix, columns=n_columns)
        n_df.to_csv(n_filepath, index=False)
        print(f"✅ n值已保存到 {n_filepath}")
        
        # 7. 保存配置参数
        self._save_config()
        
        return {
            'L_file': L_filepath,
            'E_file': E_filepath,
            'r_file': r_filepath,
            'f_file': f_filepath,
            'n_file': n_filepath
        }
    
    def _save_config(self):
        """保存配置参数"""
        config_file = os.path.join(self.output_dir, "config.txt")
        with open(config_file, 'w') as f:
            f.write("=== 配置参数 ===\n")
            f.write(f"a = {self.a}\n")
            f.write(f"U0 = {self.U0}\n")
            f.write(f"xi = {self.xi}\n")
            f.write(f"L_mu = {self.L_mu}\n")
            f.write(f"L_sigma = {self.L_sigma}\n")
            f.write(f"E_mu = {self.E_mu}\n")
            f.write(f"E_sigma = {self.E_sigma}\n")
            f.write(f"num_samples = {self.num_samples}\n")
            f.write(f"r_points = {self.r_points}\n")
            f.write(f"scan_points = {self.scan_points}\n")
        
        print(f"✅ 配置参数已保存到 {config_file}")
    
    def run(self):
        """运行完整的优化流程"""
        print("=" * 50)
        print("开始高斯分布优化计算")
        print("=" * 50)
        
        # 1. 生成样本
        samples = self.generate_gaussian_samples()
        
        if not samples:
            print("错误: 没有有效的样本！")
            return
        
        # 2. 执行优化计算
        all_results = []
        print(f"\n开始优化计算...")
        
        for i, sample in enumerate(samples):
            if (i + 1) % 10 == 0:
                print(f"  优化计算进度: {i + 1}/{len(samples)}")
            
            # 执行优化
            result = self.optimization_function(sample['L'], sample['E'])
            all_results.append(result)
        
        # 3. 保存结果
        print(f"\n保存结果...")
        saved_files = self.save_results_column_format(all_results)
        
        # 4. 输出统计信息
        self._print_statistics(all_results)
        
        print("\n" + "=" * 50)
        print("优化计算完成！")
        print("=" * 50)
        
        return all_results, saved_files
    
    def _print_statistics(self, results: List[Dict]):
        """打印统计信息"""
        print("\n=== 统计信息 ===")
        print(f"总样本数: {len(results)}")
        
        # 计算平均值
        L_values = [r['L'] for r in results]
        mean_L = np.mean(L_values)
        std_L = np.std(L_values)
        print(f"L的平均值: {mean_L:.2f} ± {std_L:.2f}")
        
        # 计算力的统计
        all_forces = []
        for result in results:
            valid_forces = result['force'][np.isfinite(result['force'])]
            all_forces.extend(valid_forces)
        
        if all_forces:
            mean_force = np.mean(all_forces)
            std_force = np.std(all_forces)
            print(f"力的平均值: {mean_force:.6f} ± {std_force:.6f}")


def main():
    """主函数"""
    # 配置参数
    config = {
        'a': 0.0,                    # a参数
        'U0': 10.0,                  # U0参数
        'xi': 30.0,                  # xi参数
        
        # L的高斯分布参数
        'L_mu': 350.0,               # L的均值
        'L_sigma': 50.0,             # L的标准差
        
        # E的高斯分布参数
        'E_mu': 10.0,                # E的均值
        'E_sigma': 3.0,           # E的标准差
        
        # 运行参数
        'num_samples': 1000,          # 总样本数
        'r_points': 2000,            # r的采样点数
        'scan_points': 1000,         # n的扫描点数
        
        # 输出目录
        'output_dir': "/home/tyt/project/Single-chain/opt+R/Rand_L+E/column_format"
    }
    
    # 创建优化器并运行
    optimizer = GaussianOptimizer(config)
    results, files = optimizer.run()
    
    return results, files


if __name__ == "__main__":
    main()
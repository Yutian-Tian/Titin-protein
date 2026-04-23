"""
基于吉布斯自由能的优化 - 自适应力值细化版本
系统：1个独立的domain
"""

import numpy as np
import pandas as pd
import os

# 参数设置
xi_f = 10.0  # 折叠态持续长度
k = 2.0      # 解折叠系数
E0 = 0.5     # 能量基准值
Ek = 5.0     # 能量系数
N = 1        # domain的数量（现在只有1个）
r_grid = 100  # r的网格数
n_grid = 10   # n的网格数
f_grid = 100  # f的初始网格数
f_max = 10.0   # 扫描力f的最大值

# 自适应细化参数
refinement_threshold = 0.5  # n变化超过此阈值时进行细化
max_refinement_level = 10    # 最大细化层级
tolerance = 1e-6       # 最小f间隔

# 设置存储路径
save_path = "/home/tyt/project/Single-chain/opt+R/Rand_xi/simulation_results/1_domain_IMS"
os.makedirs(save_path, exist_ok=True)

def energy_term_U(n_i, DeltaEi):
    """能量项: U(n_i) = ΔE_i n_i - ΔE_i cos(2π n_i)"""
    return DeltaEi * n_i - DeltaEi * np.cos(2 * np.pi * n_i)

def contour_length_Lci(n_i, xi_fi):
    """轮廓长度: L_{ci}(n_i) = ξ_fi + n_i (ξ_ui - ξ_fi)"""
    xi_ui = k * xi_fi
    return xi_fi + n_i * (xi_ui - xi_fi)

def end_to_end_factor_x_i(r_i, n_i, xi_fi):
    """端到端因子: x_i(r_i, n_i) = r_i / L_{ci}(n_i)"""
    L_ci = contour_length_Lci(n_i, xi_fi)
    return r_i / L_ci

def WLC_free_energy(x_i, L_ci):
    """WLC自由能: F_{WLC}(x_i, n_i) = (1/4) L_{ci} * [x_i^2 (3 - 2x_i) / (1 - x_i)]"""
    if x_i >= 0.999:
        return float('inf')
    return 0.25 * L_ci * (x_i**2 * (3.0 - 2.0 * x_i) / (1.0 - x_i))

def single_domain_free_energy(r_i, n_i, DeltaEi, xi_fi, f_ext):
    """单个domain的自由能: F_d(x_i, n_i) = F_{WLC}(x_i, n_i) + U(n_i) - f_ext * x_i * L_{ci}"""
    L_ci = contour_length_Lci(n_i, xi_fi)
    x_i = end_to_end_factor_x_i(r_i, n_i, xi_fi)
    F_wlc = WLC_free_energy(x_i, L_ci)
    Ui = energy_term_U(n_i, DeltaEi)
    work_term = f_ext * x_i * L_ci
    return F_wlc + Ui - work_term

def optimize_single_point(f, xi_f, tolerance=1e-8):
    """优化单个力值点对应的自由能"""
    r_max_initial = contour_length_Lci(1.0, xi_f)
    DeltaE = E0 + Ek * (xi_f - 5.0)
    
    # 初始搜索范围
    r_search_min, r_search_max = 0.0, r_max_initial
    n_search_min, n_search_max = 0.0, 1.0
    
    # 初始网格步长
    r_step = (r_search_max - r_search_min) / r_grid
    n_step = (n_search_max - n_search_min) / n_grid
    
    # 初始化最优值
    min_Fd = float('inf')
    r_best, n_best = 0.0, 0.0
    
    # 迭代细化
    while True:
        # 生成当前搜索网格
        r_vals = np.linspace(r_search_min, r_search_max, r_grid)
        n_vals = np.linspace(n_search_min, n_search_max, n_grid)
        
        # 网格搜索
        for r in r_vals:
            for n in n_vals:
                Fd = single_domain_free_energy(r, n, DeltaE, xi_f, f_ext=f)
                if Fd < min_Fd:
                    min_Fd = Fd
                    r_best, n_best = r, n
        
        # 检查收敛
        if r_step <= tolerance and n_step <= tolerance:
            break
        
        # 缩小搜索范围
        r_search_min = max(0.0, r_best - r_step)
        r_search_max = min(r_max_initial, r_best + r_step)
        n_search_min = max(0.0, n_best - n_step)
        n_search_max = min(1.0, n_best + n_step)
        
        # 更新步长
        r_step = (r_search_max - r_search_min) / 10.0
        n_step = (n_search_max - n_search_min) / 10.0
    
    x_best = end_to_end_factor_x_i(r_best, n_best, xi_f)
    return r_best, n_best, min_Fd, x_best

def detect_transition_regions(f_vals, n_opt, threshold=0.5):
    """检测r_opt发生显著变化的区域"""
    n = len(f_vals)
    # 合并相邻的过渡点
    transition_regions = []
    for i in range(n-1):
        dn = n_opt[i+1] - n_opt[i]
        if dn >= threshold:
            current_start = f_vals[i]
            current_end = f_vals[i+1]
            transition_regions.append(np.array([current_start, current_end]))
    
    return np.array(transition_regions)

def refine_force_grid(f_vals, n_opt, refinement_threshold=0.5, tolerance = 1e-10):
    """在n_opt变化剧烈的区域细化力值网格"""
    # 初始网格
    refined_f = f_vals.copy()
    refined_n = n_opt.copy()
    
    # 检测过渡区域
    transition_regions = detect_transition_regions(refined_f, refined_n, refinement_threshold)
        
    if len(transition_regions) == 0:  # 使用长度判断
        print("未检测到显著的过渡区域，停止细化")
        return f_vals.copy()  # 返回原始数据的副本
        
    print(f"检测到 {len(transition_regions)} 个跃变区域")
        
    # 收集需要细化的区间
    new_f_points = []
        
    for i in range(len(transition_regions)):  # 获取行数，代表有多少个跃变区
        f_start = transition_regions[i, 0]
        f_end = transition_regions[i, 1]
            
        # 在当前区间内插入中点
        f_mid = (f_start + f_end) / 2
            
        # 跳过已经存在的点
        if not any(abs(f_mid - f) < tolerance for f in refined_f):
            new_f_points.append(f_mid)
            print(f"  在区间 [{f_start:.6f}, {f_end:.6f}] 添加点 {f_mid:.6f}")
        
    if not new_f_points:
        print("没有需要添加的新点")
        return refined_f
        
    # 合并结果
    for f in new_f_points:
        refined_f = np.append(refined_f, f)

    # 按f值排序
    sort_idx = np.argsort(refined_f)
    refined_f = refined_f[sort_idx]
        
    print(f"细化后总点数: {len(refined_f)}")
    
    return refined_f

def optimize_with_adaptive_refinement(tolerance=1e-8, max_refinement_level=20):
    """使用自适应力值细化进行优化"""
    # 初始均匀力值网格
    f_vals_current = np.linspace(0.0, f_max, f_grid)
    
    print(f"开始初始优化 (均匀网格)...")
    print(f"力值范围: 0.0 到 {f_max}, 共 {f_grid} 个点")
    print(f"精度要求: {tolerance}, 最大细化层级: {max_refinement_level}")
    
    # 初始优化
    r_opt_current = []
    n_opt_current = []
    Fd_min_current = []
    x_opt_current = []
    
    for i, f in enumerate(f_vals_current):
        if i % 50 == 0:
            print(f"  初始优化进度: {i+1}/{len(f_vals_current)}")
        
        r_opt, n_opt, Fd_min, x_opt = optimize_single_point(f, xi_f)
        r_opt_current.append(r_opt)
        n_opt_current.append(n_opt)
        Fd_min_current.append(Fd_min)
        x_opt_current.append(x_opt)
    
    r_opt_current = np.array(r_opt_current)
    n_opt_current = np.array(n_opt_current)
    Fd_min_current = np.array(Fd_min_current)
    x_opt_current = np.array(x_opt_current)
    
    print(f"初始优化完成!")
    
    # 自适应细化循环
    for level in range(max_refinement_level):
        print(f"\n=== 细化层级 {level+1}/{max_refinement_level} ===")
        
        # 检测并细化过渡区域
        refined_f_vals = refine_force_grid(f_vals_current, n_opt_current, refinement_threshold)
        
        # 检查是否需要细化
        if len(refined_f_vals) == len(f_vals_current):
            print("没有检测到需要细化的区域，停止细化")
            break
        
        # 检查精度是否满足要求
        # 计算细化后的最小力值间隔
        if len(refined_f_vals) > 1:
            f_diffs = np.diff(np.sort(refined_f_vals))
            min_f_interval = np.min(f_diffs)
            print(f"当前最小力值间隔: {min_f_interval:.2e}, 精度要求: {tolerance:.2e}")
            
            # 修复：添加break语句
            if min_f_interval <= tolerance:
                print(f"精度已达到要求 ({tolerance:.2e})，停止细化")
                # 优化新增点后跳出循环
                break
        
        # 识别新增的力值点
        new_f_points = []
        for f in refined_f_vals:
            # 检查是否是新点
            is_new = True
            for existing_f in f_vals_current:
                if abs(f - existing_f) < 1e-10:  # 使用容差判断是否相等
                    is_new = False
                    break
            if is_new:
                new_f_points.append(f)
        
        print(f"检测到 {len(new_f_points)} 个新增力值点需要优化")
        
        if not new_f_points:
            print("没有新增点，停止细化")
            break
        
        # 优化新增点
        new_r_opt = []
        new_n_opt = []
        new_Fd_min = []
        new_x_opt = []
        
        for i, f in enumerate(new_f_points):
            if i % 10 == 0:
                print(f"  优化新增点: {i+1}/{len(new_f_points)}, f = {f:.6f}")
            
            r_opt, n_opt, Fd_min, x_opt = optimize_single_point(f, xi_f)
            new_r_opt.append(r_opt)
            new_n_opt.append(n_opt)
            new_Fd_min.append(Fd_min)
            new_x_opt.append(x_opt)
        
        # 创建从力值到结果的映射字典
        result_dict = {}
        for i, f in enumerate(f_vals_current):
            result_dict[f] = {
                'r': r_opt_current[i],
                'n': n_opt_current[i],
                'Fd': Fd_min_current[i],
                'x': x_opt_current[i]
            }
        
        # 添加新点的结果
        for i, f in enumerate(new_f_points):
            result_dict[f] = {
                'r': new_r_opt[i],
                'n': new_n_opt[i],
                'Fd': new_Fd_min[i],
                'x': new_x_opt[i]
            }
        
        # 按照refined_f_vals的顺序构建最终结果数组
        f_vals_final = refined_f_vals.copy()
        r_opt_final = []
        n_opt_final = []
        Fd_min_final = []
        x_opt_final = []
        
        for f in f_vals_final:
            if f in result_dict:
                r_opt_final.append(result_dict[f]['r'])
                n_opt_final.append(result_dict[f]['n'])
                Fd_min_final.append(result_dict[f]['Fd'])
                x_opt_final.append(result_dict[f]['x'])
            else:
                # 如果不在，重新优化（这不应该发生）
                print(f"警告: 力值点 {f} 结果缺失，重新优化...")
                r_opt, n_opt, Fd_min, x_opt = optimize_single_point(f, xi_f)
                r_opt_final.append(r_opt)
                n_opt_final.append(n_opt)
                Fd_min_final.append(Fd_min)
                x_opt_final.append(x_opt)
        
        # 转换为numpy数组
        f_vals_current = np.array(f_vals_final)
        r_opt_current = np.array(r_opt_final)
        n_opt_current = np.array(n_opt_final)
        Fd_min_current = np.array(Fd_min_final)
        x_opt_current = np.array(x_opt_final)
        
        print(f"细化层级 {level+1} 完成，总点数: {len(f_vals_current)}")
        
        # 检查是否满足精度要求
        if len(f_vals_current) > 1:
            f_diffs = np.diff(f_vals_current)
            min_f_interval = np.min(f_diffs)
            
            # 修复：添加break语句
            if min_f_interval <= tolerance:
                print(f"\n精度已达到要求 ({tolerance:.2e})，停止细化")
                break
    
    print(f"\n自适应细化完成!")
    print(f"最终力值点数: {len(f_vals_current)}")
    print(f"力值范围: [{f_vals_current.min():.6f}, {f_vals_current.max():.6f}]")
    
    return f_vals_current, r_opt_current, n_opt_current, Fd_min_current, x_opt_current

def main():

    
    # 使用自适应细化优化
    f_vals, r_opt, n_opt, Fd_min, x_opt = optimize_with_adaptive_refinement()
    
    print(f"\n优化完成! 总点数: {len(f_vals)}")
    print(f"力值范围: [{f_vals.min():.6f}, {f_vals.max():.6f}]")
    print(f"r_opt范围: [{r_opt.min():.6f}, {r_opt.max():.6f}]")
    
    
    # 保存结果到CSV文件
    print(f"\n正在保存结果...")
    results_df = pd.DataFrame({
        'f': f_vals,
        'r_opt': r_opt,
        'n_opt': n_opt,
        'Fd_min': Fd_min,
        'x_opt': x_opt
    })
    
    results_df.to_csv(os.path.join(save_path, "single_domain_results_refined.csv"), index=False)
    
    # 保存参数
    DeltaE = E0 + Ek * (xi_f - 5.0)
    params_df = pd.DataFrame({
        'Parameter': ['xi_f', 'E0', 'Ek', 'k', 'DeltaE', 'r_grid', 'n_grid', 
                     'f_grid_initial', 'f_max', 'refinement_threshold', 
                     'max_refinement_level', 'min_f_interval', 'total_points'],
        'Value': [xi_f, E0, Ek, k, DeltaE, r_grid, n_grid, 
                 f_grid, f_max, refinement_threshold, 
                 max_refinement_level, tolerance, len(f_vals)]
    })
    params_df.to_csv(os.path.join(save_path, "domain_parameters_refined.csv"), index=False)
      
    
    print(f"\n所有结果已保存到 '{save_path}' 目录")
    print(f"生成的文件:")
    print(f"  - single_domain_results_refined.csv: 细化后的结果")
    print(f"  - domain_parameters_refined.csv: 细化参数")
    
    return f_vals, r_opt, n_opt, Fd_min, x_opt

# 主程序入口
if __name__ == "__main__":
    results = main()
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <limits>

// 系统参数 (以k_B T为单位)
const double L = 200.0;          // 总长度
const int N = 5;          // 最大n值
const double xi = 30.0;           // ξ参数
const double a = 1.0;            // a参数
const double kG = 1.0;           // 弹簧常数
const double deltaE1 = 50.0;      // ΔE1
const double deltaE2 = 5.0;      // ΔE2
const double U0 = 15.0;           // U0

// 根据条件计算r2
double calculate_r2(double r1, double f) {
    // 3*k_G*(r2 - r1) = f
    // r2 = r1 + f/(3*k_G)
    return r1 + f / (3.0 * kG);
}

// WLC自由能函数 (k_B T = 1)
double F_WLC(double x, double Ln) {
    if (x >= 1.0 || x <= -1.0 || Ln <= 0) {
        return std::numeric_limits<double>::max();
    }
    
    double term1 = (M_PI * M_PI) / (2.0 * Ln) * (1.0 - x * x);
    double term2 = (2.0 * Ln) / (M_PI *   (1.0 - x * x));
    
    return term1 + term2;  // 不再乘以k_B T
}

// U(n)函数
double U_n(double n) {
    return deltaE1 * n + deltaE2 * n * n - U0 * std::cos(2.0 * M_PI * n);
}

// 高斯自由能函数 (k_B T = 1)
double F_G(double r, double k) {
    return (3.0 / 2.0) * k * r * r;  // 不再乘以k_B T
}

// 总自由能函数 F(r1, n, f)
double total_F(double r1, double n, double f) {
    // 检查约束条件
    if (n < 0 || n > N || r1 < 0) {
        return std::numeric_limits<double>::max();
    }
    
    double L_n = L - (N - n) * xi;
    if (r1 > L_n) {
        return std::numeric_limits<double>::max();
    }
    
    // 计算r2
    double r2 = calculate_r2(r1, f);
    
    // 计算x_n
    double x_n = (r2 - (N - n) * a) / L_n;
    
    // 计算各项自由能
    double F_wlc = F_WLC(x_n, L_n);
    double U_val = U_n(n);
    double F_g = F_G(r2 - r1, kG);
    double work_term = -f * r2;
    
    return F_wlc + U_val + F_g + work_term;
}

// 网格搜索优化函数
void minimize_F(double f, double& best_r1, double& best_r2, double& best_n, double& min_F) {
    min_F = std::numeric_limits<double>::max();
    
    // 网格搜索参数
    const int n_steps = 1000;  // 增加步数提高精度
    const int r1_steps = 1000;
    const double r1_max = L;  // 最大r1值
    
    for (int i_n = 0; i_n <= n_steps; ++i_n) {
        double n = (N * i_n) / n_steps;
        
        double L_n = L - (N - n) * xi;
        double r1_max_current = std::min(r1_max, L_n);
        
        for (int i_r1 = 0; i_r1 <= r1_steps; ++i_r1) {
            double r1 = (r1_max_current * i_r1) / r1_steps;
            
            double current_F = total_F(r1, n, f);
            
            if (current_F < min_F) {
                min_F = current_F;
                best_r1 = r1;
                best_n = n;
                best_r2 = calculate_r2(r1, f);
            }
        }
        
        // 显示进度
        if (i_n % 50 == 0) {
            std::cout << "f = " << f << ", 进度: " << (100.0 * i_n / n_steps) << "%" << std::endl;
        }
    }
}

int main() {
    // 打开输出文件
    std::ofstream output_file("WLC+Gauss+force_C_results.csv");
    if (!output_file.is_open()) {
        std::cerr << "无法打开输出文件!" << std::endl;
        return 1;
    }
    
    // 写入CSV文件头
    output_file << "f,r1,r2,n,F_min\n";
    
    // f从0到10进行扫描
    const int f_steps = 1000;
    
    std::cout << "开始最小化计算..." << std::endl;
    std::cout << "参数信息 (以k_B T为单位):" << std::endl;
    std::cout << "L = " << L << ", N = " << N << ", xi = " << xi << std::endl;
    std::cout << "ΔE1 = " << deltaE1 << ", ΔE2 = " << deltaE2 << ", U0 = " << U0 << std::endl;
    
    for (int i = 0; i <= f_steps; ++i) {
        double f = (10.0 * i) / f_steps;
        
        double best_r1, best_r2, best_n, min_F;
        minimize_F(f, best_r1, best_r2, best_n, min_F);
        
        // 输出到文件
        output_file << f << "," << best_r1 << "," << best_r2 << "," << best_n << "," << min_F << "\n";
        
        // 进度显示
        if (i % 100 == 0) {
            std::cout << "f = " << f << ": r1 = " << best_r1 << ", r2 = " << best_r2 
                      << ", n = " << best_n << ", F_min = " << min_F << std::endl;
        }
    }
    
    output_file.close();
    std::cout << "计算完成! 结果已保存到 WLC+Gauss+force_C_results.csv" << std::endl;
    
    // 创建参数文件记录使用的参数
    std::ofstream param_file("parameters_used.txt");
    param_file << "使用的参数 (以k_B T为单位):\n";
    param_file << "L = " << L << "\n";
    param_file << "N = " << N << "\n";
    param_file << "xi = " << xi << "\n";
    param_file << "kG = " << kG << "\n";
    param_file << "a = " << a << "\n";
    param_file << "ΔE1 = " << deltaE1 << "\n";
    param_file << "ΔE2 = " << deltaE2 << "\n";
    param_file << "U0 = " << U0 << "\n";
    param_file.close();
    
    return 0;
}
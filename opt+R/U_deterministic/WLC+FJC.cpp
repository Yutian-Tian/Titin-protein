#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

using namespace std;

// 物理常数和参数 (k_B T = l_p = 1)
const double L = 200.0;    // 总长度
const double xi = 30.0;    // 参数
const double a = 3.0;      // Kuhn长度
const double N_max = 5.0;  // n的最大值 (连续值)

// WLC自由能
double F_WLC(double r1, double n) {
    double L_n = L - (N_max - n) * xi;
    if (L_n <= 1e-10) return 1e20; // 避免无效值
    double x_n = r1 / L_n;
    if (x_n >= 1.0) return 1e20; // 避免无效值
    
    double term1 = (M_PI * M_PI) / (2.0 * L_n) * (1.0 - x_n * x_n);
    double term2 = (2.0 * L_n) / (M_PI * (1.0 - x_n * x_n));
    
    return term1 + term2;
}

// Langevin函数
double langevin(double x) {
    if (fabs(x) < 1e-10) {
        return x / 3.0 - x * x * x / 45.0;
    }
    return 1.0 / tanh(x) - 1.0 / x;
}

// Langevin反函数 (数值求解)
double inv_langevin(double y) {
    if (y >= 0.999) return 10.0;
    if (y <= 0.0) return 0.0;
    
    // 使用牛顿法求解
    double x = 3.0 * y; // 初始猜测
    double tolerance = 1e-10;
    int max_iter = 100;
    
    for (int i = 0; i < max_iter; i++) {
        double L_val = langevin(x);
        double L_deriv = (1.0 / (x * x)) - 1.0 / (sinh(x) * sinh(x));
        
        // 避免除零
        if (fabs(L_deriv) < 1e-15) break;
        
        double delta = (y - L_val) / L_deriv;
        x += delta;
        if (fabs(delta) < tolerance) break;
    }
    return x;
}

// FJC自由能 (通过积分力得到)
double F_FJC(double r, double n) {
    if (n >= N_max - 1e-10) return 0; // n接近N_max时FJC部分消失
    if (r <= 0) return 0;
    
    double max_extension = (N_max - n) * a;
    if (r >= max_extension) return 1e20;
    
    // 数值积分
    int steps = 100;
    double integral = 0.0;
    double dr = r / steps;
    
    for (int i = 0; i < steps; i++) {
        double r_current = (i + 0.5) * dr;
        double strain = r_current / max_extension;
        
        double x = inv_langevin(strain);
        double force = x; // f * a / kT = x
        integral += force * dr;
    }
    
    return integral;
}

// 总自由能
double total_free_energy(double r1, double r2, double n, double deltaE1, double deltaE2, double U0) {
    // 约束条件
    if (n < 0 || n > N_max) return 1e20;
    if (r1 < 0 || r2 < 0) return 1e20;
    if (r1 > r2) return 1e20;
    
    double L_n = L - (N_max - n) * xi;
    if (L_n <= 0) return 1e20;
    if (r1 > L_n) return 1e20;
    
    double F_wlc = F_WLC(r1, n);
    double F_fjc = F_FJC(r2 - r1, n);
    double energy_terms = deltaE1 * n + deltaE2 * n * n - U0 * cos(2 * M_PI * n);
    
    return F_wlc + energy_terms + F_fjc;
}

// 使用均匀扫描寻找最小值
void uniform_scan_minimize(double r2, double deltaE1, double deltaE2, double U0,
                          double& best_r1, double& best_n, double& min_energy) {
    min_energy = 1e20;
    best_r1 = 0;
    best_n = 0;
    
    // 扫描参数
    const int n_steps = 100;  // n的扫描步数
    const int r1_steps = 100; // r1的扫描步数
    
    for (int i_n = 0; i_n <= n_steps; i_n++) {
        double n = (i_n * N_max) / n_steps;
        
        double L_n = L - (N_max - n) * xi;
        if (L_n <= 0) continue;
        
        for (int i_r1 = 0; i_r1 <= r1_steps; i_r1++) {
            double r1 = (i_r1 * min(r2, L_n)) / r1_steps;
            
            // 检查约束条件
            if (r1 > r2) continue;
            
            double energy = total_free_energy(r1, r2, n, deltaE1, deltaE2, U0);
            
            if (energy < min_energy) {
                min_energy = energy;
                best_r1 = r1;
                best_n = n;
            }
        }
    }
}

// 计算张力 (数值导数)
double calculate_tension(double r2, double deltaE1, double deltaE2, double U0, double dr = 1e-5) {
    double r1_plus, n_plus, energy_plus;
    double r1_minus, n_minus, energy_minus;
    
    uniform_scan_minimize(r2 + dr, deltaE1, deltaE2, U0, r1_plus, n_plus, energy_plus);
    uniform_scan_minimize(r2 - dr, deltaE1, deltaE2, U0, r1_minus, n_minus, energy_minus);
    
    return (energy_plus - energy_minus) / (2 * dr);
}

int main() {
    // 参数设置
    double deltaE1 = 10.0;
    double deltaE2 = 0.0;
    double U0 = 20.0;
    
    // 输出文件
    ofstream outfile("WLC+FJC_results.csv");
    outfile << "r2,r1,n,free_energy,tension" << endl;
    outfile << fixed << setprecision(8);
    
    // r2范围 [0, 200]
    int r2_steps = 1000;
    
    cout << "开始计算 (均匀扫描方法)..." << endl;
    cout << "参数: L=" << L << ", xi=" << xi << ", a=" << a << ", N_max=" << N_max << endl;
    cout << "扫描精度: n_steps=100, r1_steps=100" << endl;
    
    for (int i = 0; i <= r2_steps; i++) {
        double r2 = (i * 200.0) / r2_steps;
        
        double best_r1, best_n, min_energy;
        uniform_scan_minimize(r2, deltaE1, deltaE2, U0, best_r1, best_n, min_energy);
        
        double tension = calculate_tension(r2, deltaE1, deltaE2, U0);
        
        outfile << r2 << "," << best_r1 << "," << best_n << "," 
                << min_energy << "," << tension << endl;
        
        if (i % 20 == 0) {
            cout << "进度: " << (i * 100 / r2_steps) << "%, r2=" << r2 
                 << ", r1=" << best_r1 << ", n=" << best_n 
                 << ", F=" << min_energy << ", f=" << tension << endl;
        }
    }
    
    outfile.close();
    cout << "计算完成！结果已保存到 WLC+FJC_results.csv" << endl;
    cout << "输出格式: r2, r1, n, free_energy, tension" << endl;
    
    return 0;
}
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
using namespace std;

// 无量纲单位: k_B T = l_p = 1
const double a = 0.0;           // 无量纲长度
const double L = 200.0;         // 总长度
const double xi = 30.0;         // ξ 参数
const int N = 5;                // n的最大值
const double DeltaE1 = 50.0;    // ΔE1 (以kT为单位)
const double DeltaE2 = 10.0;     // ΔE2 (以kT为单位)s
const double U0 = 10.0;          // U0 (以kT为单位)

// WLC 自由能 (简化版本，l_p = 1)
double F_WLC(double R, double n) {
    double L_n = L - (N - n) * xi;
    if (L_n <= 0) return numeric_limits<double>::max();
    
    double x_n = (R - (N - n) * a) / L_n;
    if (x_n < 0.0 || x_n > 1.0) return numeric_limits<double>::max();
    
    double term1 = (M_PI * M_PI) / (2.0 * L_n) * (1.0 - x_n * x_n);
    double term2 = (2.0 * L_n) / (M_PI * (1.0 - x_n * x_n));
    
    return term1 + term2;  // 直接返回，l_p = 1已隐含
}

// U(n) 函数
double U_n(double n) {
    return DeltaE1 * n + DeltaE2 * n * n - U0 * cos(2.0 * M_PI * n);
}

// 总自由能 F(R, n)
double F_total(double R, double n, double f) {
    // 检查约束条件
    double L_n = L - (N - n) * xi;
    if (n < 0 || n > N || R < (N - n) * a || R > L_n) {
        return numeric_limits<double>::max();
    }
    
    return F_WLC(R, n) + U_n(n) - f * R;  // f 也是无量纲力
}

// 在给定f下寻找最小F(R,n)
void minimize_F(double f, double& best_R, double& best_n, double& min_F) {
    min_F = numeric_limits<double>::max();
    best_R = 0;
    best_n = 0;
    
    // 离散化搜索空间
    int n_steps = 1000;
    int R_steps = 10000;
    
    for (int i = 0; i <= n_steps; i++) {
        double n = (double)i / n_steps * N;
        
        double L_n = L - (N - n) * xi;
        if (L_n <= 0) continue;
        
        for (int j = 0; j <= R_steps; j++) {
            double R = (double)j / R_steps * L;
            
            double F_val = F_total(R, n, f);
            
            if (F_val < min_F) {
                min_F = F_val;
                best_R = R;
                best_n = n;
            }
        }
    }
}

// 计算张力 (简化版本)
double tension(double R, double n, double f) {
    double L_n = L - (N - n) * xi;
    double x_n = (R - (N - n) * a) / L_n;
    
    // 添加数值稳定性检查
    if (1.0 - x_n * x_n < 1e-15) {
        return 0.0;
    }
    
    double term1 = -M_PI * M_PI * x_n / (L_n * L_n);
    double term2 = 4.0 * x_n / (M_PI * (1.0 - x_n * x_n) * (1.0 - x_n * x_n));
    
    return term1 + term2;
}

int main() {
    // 打开输出文件
    ofstream outfile("WLC+force_results.csv");
    if (!outfile) {
        cerr << "无法创建输出文件!" << endl;
        return 1;
    }
    
    // 写入CSV头部
    outfile << "f,R_opt,n_opt,F_min,tension" << endl;
    
    // f的取值范围：0到10 (无量纲力)
    int f_steps = 1000;
    double f_max = 10.0;
    
    cout << "开始最小化计算 (使用无量纲单位 kT = lp = 1)..." << endl;
    cout << "参数: L=" << L << ", xi=" << xi << ", N=" << N << ", DeltaE1=" << DeltaE1 << endl;
    
    for (int i = 0; i <= f_steps; i++) {
        double f = (double)i / f_steps * f_max;
        
        double best_R, best_n, min_F;
        minimize_F(f, best_R, best_n, min_F);
        
        double tens = tension(best_R, best_n, f);
        
        // 输出到文件
        outfile << f << "," << best_R << "," << best_n << "," << min_F << "," << tens << endl;
        
        // 进度显示
        if (i % 100 == 0) {
            cout << "f = " << f << ": R_opt = " << best_R << ", n_opt = " << best_n 
                 << ", F_min = " << min_F << ", tension = " << tens << endl;
        }
    }
    
    outfile.close();
    cout << "计算完成! 结果已保存到 WLC+force_results.csv" << endl;
    
    return 0;
}
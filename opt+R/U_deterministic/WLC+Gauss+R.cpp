#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <limits>

// 参数设置
const double L = 200.0;
const double xi = 30.0;
const double a = 3.0;
const int N = 5;
const double deltaE1 = 10.0;
const double U0 = 20.0;
const double deltaE2 = 0.0;

const int Rsteps = 10000;
const int nsteps = 1000;
const int rsteps = 1000;

// 结果结构体
struct Result {
    double R;
    double n_u;
    double r;
    double F;
    double dF_dR;
};

class PolymerModel {
private:
    // 防止除零错误的小值
    const double epsilon = 1e-10;
    
public:
    double F_WLC(double R, double r, double n_u) {
        double Ln_val = L - (N - n_u) * xi;
        
        // 避免除零错误
        if (Ln_val <= epsilon) {
            return std::numeric_limits<double>::infinity();
        }
        
        double xn_val = (R - r) / Ln_val;
        
        // 避免1-x_n^2为负数或接近零
        if (std::abs(xn_val) >= 1.0 || std::abs(1.0 - xn_val*xn_val) < 0) {
            return std::numeric_limits<double>::infinity();
        }
        
        double term1 = (M_PI * M_PI) / (2.0 * Ln_val) * (1.0 - xn_val*xn_val);
        double term2 = (2.0 * Ln_val) / (M_PI * (1.0 - xn_val*xn_val));
        
        return term1 + term2;
    }
    
    double F_FJC(double r, double n_u) {
        if (n_u >= N) {
            return std::numeric_limits<double>::infinity();
        }
        double denominator = 2.0 * (N - n_u) * a * a + epsilon;
        return (3.0 * r * r) / denominator + 1.5 * std::log(N - n_u + epsilon);
    }
    
    double total_F(double R, double r, double n_u, double delta_E1 = deltaE1, double delta_E2 = deltaE2) {
        return F_WLC(R, r, n_u) + delta_E1 * n_u + delta_E2 * n_u * n_u - U0 * std::cos(2.0 * M_PI * n_u) + F_FJC(r, n_u);
    }
    
    double dF_dR(double R, double r, double n_u) {
        double Lc = L - (N - n_u) * xi;
        double x_val = (R - r) / Lc;
        
        double d_term1_dr = -M_PI * M_PI * x_val / (Lc * Lc);
        double d_term2_dr = 4.0 * x_val / (M_PI * (1.0 - x_val * x_val));
        
        return d_term1_dr + d_term2_dr;
    }
    
    bool check_constraints(double R, double r, double n_u) {
        // 约束1: 0 <= r <= n_u * a
        if (!(0.0 <= r && r <= n_u * a)) {
            return false;
        }
        
        // 约束2: 0 <= R - r <= L - (N - n_u) * xi
        double Ln_val = L - (N - n_u) * xi;
        if (Ln_val <= epsilon) {
            return false;
        }
        
        double R_minus_r = R - r;
        if (!(0.0 <= R_minus_r && R_minus_r <= Ln_val)) {
            return false;
        }
        
        return true;
    }
    
    std::vector<Result> find_optimal_parameters(const std::vector<double>& R_values) {
        std::vector<Result> results;
        
        for (size_t i = 0; i < R_values.size(); ++i) {
            double R = R_values[i];
            double min_F = std::numeric_limits<double>::infinity();
            double best_n_u = 0.0;
            double best_r = 0.0;
            double best_F = 0.0;
            
            // 扫描所有可能的n_u值
            for (int j = 0; j < nsteps; ++j) {
                double n_u = j * (N - epsilon) / (nsteps - 1);
                
                // 计算r的最大值
                double r_max = std::min(n_u * a, R);
                if (r_max <= epsilon) {
                    continue;
                }
                
                // 扫描r值
                for (int k = 0; k < rsteps; ++k) {
                    double r = k * r_max / (rsteps - 1);
                    
                    // 检查约束条件
                    if (!check_constraints(R, r, n_u)) {
                        continue;
                    }
                    
                    // 计算当前参数下的F值
                    double current_F = total_F(R, r, n_u);
                    
                    // 更新最小值
                    if (current_F < min_F) {
                        min_F = current_F;
                        best_n_u = n_u;
                        best_r = r;
                        best_F = current_F;
                    }
                }
            }
            
            // 如果没有找到满足约束的解，使用边界值
            if (min_F == std::numeric_limits<double>::infinity()) {
                best_n_u = 0.0;
                best_r = 0.0;
                best_F = total_F(R, 0.0, 0.0);
            }
            
            // 计算∂F/∂R
            double dF_dR_val = dF_dR(R, best_r, best_n_u);
            
            Result result;
            result.R = R;
            result.n_u = best_n_u;
            result.r = best_r;
            result.F = best_F;
            result.dF_dR = dF_dR_val;
            
            results.push_back(result);
            
            // 显示进度
            if ((i + 1) % 100 == 0) {
                std::cout << "处理进度: " << (i + 1) << "/" << R_values.size() 
                         << ", R=" << R << ", r=" << best_r 
                         << ", 最优nu=" << best_n_u << ", 最小F=" << best_F 
                         << ", ∂F/∂R=" << dF_dR_val << std::endl;
            }
        }
        
        return results;
    }
    
    void save_results(const std::vector<Result>& results, const std::string& filename) {
        std::ofstream file(filename);
        file << "R,n_u,r,F,dF_dR\n";
        
        for (const auto& result : results) {
            file << result.R << "," << result.n_u << "," << result.r << "," 
                 << result.F << "," << result.dF_dR << "\n";
        }
        
        file.close();
    }
    
    void print_statistics(const std::vector<Result>& results) {
        if (results.empty()) return;
        
        double min_R = results[0].R, max_R = results[0].R;
        double min_F = results[0].F, max_F = results[0].F;
        double min_n_u = results[0].n_u, max_n_u = results[0].n_u;
        double min_r = results[0].r, max_r = results[0].r;
        
        for (const auto& result : results) {
            min_R = std::min(min_R, result.R);
            max_R = std::max(max_R, result.R);
            min_F = std::min(min_F, result.F);
            max_F = std::max(max_F, result.F);
            min_n_u = std::min(min_n_u, result.n_u);
            max_n_u = std::max(max_n_u, result.n_u);
            min_r = std::min(min_r, result.r);
            max_r = std::max(max_r, result.r);
        }
        
        std::cout << "\n统计信息:\n";
        std::cout << "R范围: " << min_R << " 到 " << max_R << "\n";
        std::cout << "F范围: " << min_F << " 到 " << max_F << "\n";
        std::cout << "n_u范围: " << min_n_u << " 到 " << max_n_u << "\n";
        std::cout << "r范围: " << min_r << " 到 " << max_r << "\n";
        
        // 验证约束条件
        std::cout << "\n约束条件验证:\n";
        std::vector<int> indices = {0, static_cast<int>(results.size()/4), 
                                   static_cast<int>(results.size()/2), 
                                   static_cast<int>(3*results.size()/4), 
                                   static_cast<int>(results.size()-1)};
        
        for (int idx : indices) {
            if (idx < results.size()) {
                const auto& row = results[idx];
                bool constraint1 = (0.0 <= row.r) && (row.r <= row.n_u * a);
                double Ln_val = L - (N - row.n_u) * xi;
                bool constraint2 = (0.0 <= (row.R - row.r)) && ((row.R - row.r) <= Ln_val);
                
                std::cout << "R=" << row.R << ": r=" << row.r << " in [0, " << row.n_u*a 
                         << "]? " << (constraint1 ? "true" : "false") 
                         << ", R-r=" << (row.R - row.r) << " in [0, " << Ln_val 
                         << "]? " << (constraint2 ? "true" : "false") << std::endl;
            }
        }
    }
};

int main() {
    PolymerModel model;
    
    // 生成R值范围
    std::vector<double> R_values;
    for (int i = 0; i < Rsteps; ++i) {
        R_values.push_back(i * (L - 0.01) / (Rsteps - 1));
    }
    
    // 进行计算
    std::cout << "开始计算..." << std::endl;
    std::vector<Result> results = model.find_optimal_parameters(R_values);
    std::cout << "计算完成!" << std::endl;
    
    // 保存结果
    model.save_results(results, "WLC+Gauss_results.csv");
    std::cout << "结果已保存到 WLC+Gauss_results.csv" << std::endl;
    
    // 输出统计信息
    model.print_statistics(results);
    
    return 0;
}




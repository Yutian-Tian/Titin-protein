#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <filesystem>
#include <string>
#include <sstream>
#include <tuple>

using namespace std;

// 全局参数设置
const int num_samples = 300;
const double k = 2.0;          // ξ_u = k * ξ_f
const double Ek = 1.0;         // 能量缩放系数
const double Uk = 3.0;         // 周期势缩放系数
const int n_grid_points = 10000; // n的均匀划分点数
const int r_grid_points = 1000; // r值点数
const double mu = 10.0;
const double sigma = 10.0;
// 修改常量名称，避免与标准库函数冲突
const double lower_bound_val = 5.0;
const double upper_bound_val = 15.0;
const double PI = 3.14159265358979323846;

// 指定输出路径
const string output_dir = "/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results";

// WLC自由能函数 (k_B T和l_p设为1，不显式出现在公式中)
double F_WLC(double x, double L_c) {
    // 避免除以零或接近零
    if (x >= 1.0 || x <= -1.0 || L_c <= 0.0) {
        return numeric_limits<double>::infinity();
    }
    
    double term1 = (PI * PI) * (1.0 - x * x)/ (2.0 * L_c);
    double term2 = (2.0 * L_c) / (PI * (1.0 - x * x));
    return term1 + term2;
}

// U(n)函数
double U_n(double n, double xi_u) {
    double delta_E = Ek * pow(xi_u, 1);
    double U0 = Uk * pow(xi_u, 1);
    return delta_E * n - U0 * cos(2.0 * PI * n);
}

// 总自由能函数
double total_free_energy(double n, double r, double xi_f, double xi_u) {
    // 计算轮廓长度
    double L_c = xi_f + n * (xi_u - xi_f);
    
    // 计算x
    double x = r / L_c;
    
    // 计算自由能
    double F_wlc = F_WLC(x, L_c);
    double U = U_n(n, xi_u);
    
    // 检查是否是无穷大
    if (F_wlc == numeric_limits<double>::infinity()) {
        return numeric_limits<double>::infinity();
    }
    
    return F_wlc + U;
}

// 力的计算函数 (k_B T和l_p设为1，不显式出现在公式中)
double calculate_force(double r, double n, double xi_f, double xi_u) {
    // 计算轮廓长度
    double L_c = xi_f + n * (xi_u - xi_f);
    
    if (L_c <= 0.0) {
        return numeric_limits<double>::quiet_NaN();
    }
    
    // 计算x
    double x = r / L_c;
    
    // 检查x是否在有效范围内
    if (x >= 1.0 || x <= -1.0) {
        return numeric_limits<double>::quiet_NaN();
    }
    
    // 根据公式计算力
    double term1 = - (PI * PI * x) / (L_c * L_c);
    double term2 = (4.0 * x) / (PI * (1.0 - x * x) * (1.0 - x * x));
    
    return term1 + term2;
}

// 为单个采样执行优化
tuple<vector<double>, vector<double>, vector<double>> 
optimize_for_sample(double xi_f, double xi_u, const vector<double>& n_grid) {
    // 为采样生成r值范围：从0到0.99*ξ_u
    double r_max = 0.95 * xi_u;
    vector<double> r_values(r_grid_points);
    for (int i = 0; i < r_grid_points; ++i) {
        r_values[i] = (i * r_max) / (r_grid_points - 1);
    }
    
    // 存储当前采样的结果
    vector<double> current_n(r_grid_points);
    vector<double> current_f(r_grid_points);
    
    // 对每个r值进行优化
    for (int i = 0; i < r_grid_points; ++i) {
        double r = r_values[i];
        
        // 初始化最小自由能和对应的n
        double min_energy = numeric_limits<double>::infinity();
        double best_n = 0.5;
        
        // 均匀扫描n的可行域
        for (double n : n_grid) {
            // 计算自由能
            double energy = total_free_energy(n, r, xi_f, xi_u);
            
            // 更新最小值
            if (energy < min_energy) {
                min_energy = energy;
                best_n = n;
            }
        }
        
        // 使用找到的最佳n计算力
        double n_opt = best_n;
        double f_opt = calculate_force(r, n_opt, xi_f, xi_u);
        
        current_n[i] = n_opt;
        current_f[i] = f_opt;
    }
    
    return make_tuple(r_values, current_n, current_f);
}

// 保存结果到CSV文件
void save_results(const vector<vector<double>>& all_r,
                  const vector<vector<double>>& all_n,
                  const vector<vector<double>>& all_f,
                  const vector<double>& xi_f_samples,
                  const vector<double>& xi_u_samples) {
    // 确保输出目录存在
    filesystem::create_directories(output_dir);
    
    // 保存r值数据
    ofstream r_file(output_dir + "/r_values.csv");
    for (int i = 0; i < all_r[0].size(); ++i) {
        for (int j = 0; j < all_r.size(); ++j) {
            r_file << all_r[j][i];
            if (j < all_r.size() - 1) r_file << ",";
        }
        r_file << "\n";
    }
    r_file.close();
    
    // 保存n值数据
    ofstream n_file(output_dir + "/n_values.csv");
    for (int i = 0; i < all_n[0].size(); ++i) {
        for (int j = 0; j < all_n.size(); ++j) {
            n_file << all_n[j][i];
            if (j < all_n.size() - 1) n_file << ",";
        }
        n_file << "\n";
    }
    n_file.close();
    
    // 保存f值数据
    ofstream f_file(output_dir + "/f_values.csv");
    for (int i = 0; i < all_f[0].size(); ++i) {
        for (int j = 0; j < all_f.size(); ++j) {
            f_file << all_f[j][i];
            if (j < all_f.size() - 1) f_file << ",";
        }
        f_file << "\n";
    }
    f_file.close();
    
    // 保存参数信息
    ofstream params_file(output_dir + "/parameters.csv");
    params_file << "xi_f,xi_u,r_max\n";
    for (int i = 0; i < xi_f_samples.size(); ++i) {
        params_file << xi_f_samples[i] << ","
                   << xi_u_samples[i] << ","
                   << 0.95 * xi_u_samples[i] << "\n";
    }
    params_file.close();
    
    // 保存配置信息
    ofstream config_file(output_dir + "/config.csv");
    config_file << "parameter,value,description\n";
    config_file << "num_samples," << num_samples << ",采样数量\n";
    config_file << "k," << k << ",ξ_u = k * ξ_f 的系数\n";
    config_file << "Ek," << Ek << ",能量缩放系数 (delta_E = Ek * ξ_u^3)\n";
    config_file << "Uk," << Uk << ",周期势缩放系数 (U0 = Uk * ξ_u^3)\n";
    config_file << "n_grid_points," << n_grid_points << ",n的均匀划分点数\n";
    config_file << "num_r_points," << r_grid_points << ",每个采样的r值点数\n";
    config_file.close();
    
    cout << "\n结果已保存到CSV文件:\n";
    cout << "- r_values.csv: " << output_dir << "/r_values.csv" 
              << " (形状: " << all_r.size() << "x" << all_r[0].size() << ")\n";
    cout << "- n_values.csv: " << output_dir << "/n_values.csv" 
              << " (形状: " << all_n.size() << "x" << all_n[0].size() << ")\n";
    cout << "- f_values.csv: " << output_dir << "/f_values.csv" 
              << " (形状: " << all_f.size() << "x" << all_f[0].size() << ")\n";
    cout << "\n参数信息已保存到: " << output_dir << "/parameters.csv\n";
    cout << "配置信息已保存到: " << output_dir << "/config.csv\n";
}

// 采样ξ_f，只保留在指定范围内的样本
tuple<vector<double>, vector<double>> 
sample_xi_f_with_constraint(int num_samples, double min_val, double max_val) {
    vector<double> xi_f_samples;
    vector<double> xi_u_samples;
    
    random_device rd;
    mt19937 gen(rd());
    gen.seed(42);
    normal_distribution<> dist(mu, sigma);
    
    cout << "正在采样ξ_f，要求范围: [" << min_val << ", " << max_val << "]\n";
    
    // 继续采样直到达到所需数量
    while (xi_f_samples.size() < num_samples) {
        int batch_size = min(num_samples * 2, num_samples - (int)xi_f_samples.size() + 100);
        
        for (int i = 0; i < batch_size; ++i) {
            double xi_f = dist(gen);
            
            // 筛选符合条件的样本
            if (xi_f >= min_val && xi_f <= max_val) {
                xi_f_samples.push_back(xi_f);
                xi_u_samples.push_back(k * xi_f);
                
                if (xi_f_samples.size() >= num_samples) {
                    break;
                }
            }
        }
    }
    
    // 截取所需数量的样本
    xi_f_samples.resize(num_samples);
    xi_u_samples.resize(num_samples);
    
    return make_tuple(xi_f_samples, xi_u_samples);
}

// 计算统计信息
void print_stats(const vector<double>& data, const string& name) {
    if (data.empty()) return;
    
    double sum = 0.0;
    double min_val = numeric_limits<double>::max();
    double max_val = numeric_limits<double>::lowest();
    int valid_count = 0;
    
    for (const auto& val : data) {
        if (!isnan(val)) {
            sum += val;
            min_val = min(min_val, val);
            max_val = max(max_val, val);
            valid_count++;
        }
    }
    
    double mean = (valid_count > 0) ? sum / valid_count : 0.0;
    
    // 计算标准差
    double std_dev = 0.0;
    if (valid_count > 1) {
        double variance = 0.0;
        for (const auto& val : data) {
            if (!isnan(val)) {
                variance += (val - mean) * (val - mean);
            }
        }
        std_dev = sqrt(variance / valid_count);
    }
    
    cout << name << "统计:\n";
    cout << "  有效点数: " << valid_count << "/" << data.size() << "\n";
    cout << "  范围: [" << min_val << ", " << max_val << "]\n";
    cout << "  均值: " << mean << "\n";
    cout << "  标准差: " << std_dev << "\n";
}

int main() {
    cout << string(60, '=') << "\n";
    cout << "开始执行单链自由能优化计算\n";
    cout << "采样数: " << num_samples << "\n";
    cout << "参数: k=" << k << ", Ek=" << Ek << ", Uk=" << Uk << "\n";
    cout << "网格: n_grid_points=" << n_grid_points 
              << ", r_grid_points=" << r_grid_points << "\n";
    cout << string(60, '=') << "\n";
    
    // Step 1: 采样 ξ_f
    cout << "\nStep 1: 采样" << num_samples << "个ξ_f...\n";
    auto [xi_f_samples, xi_u_samples] = sample_xi_f_with_constraint(num_samples, lower_bound_val, upper_bound_val);
    
    // 计算ξ_f和ξ_u的统计信息
    double xi_f_sum = 0.0, xi_u_sum = 0.0;
    double xi_f_min = numeric_limits<double>::max(), xi_f_max = numeric_limits<double>::lowest();
    double xi_u_min = numeric_limits<double>::max(), xi_u_max = numeric_limits<double>::lowest();
    
    for (int i = 0; i < num_samples; ++i) {
        xi_f_sum += xi_f_samples[i];
        xi_u_sum += xi_u_samples[i];
        xi_f_min = min(xi_f_min, xi_f_samples[i]);
        xi_f_max = max(xi_f_max, xi_f_samples[i]);
        xi_u_min = min(xi_u_min, xi_u_samples[i]);
        xi_u_max = max(xi_u_max, xi_u_samples[i]);
    }
    
    double xi_f_mean = xi_f_sum / num_samples;
    double xi_u_mean = xi_u_sum / num_samples;
    
    double xi_f_std = 0.0, xi_u_std = 0.0;
    for (int i = 0; i < num_samples; ++i) {
        xi_f_std += pow(xi_f_samples[i] - xi_f_mean, 2);
        xi_u_std += pow(xi_u_samples[i] - xi_u_mean, 2);
    }
    xi_f_std = sqrt(xi_f_std / num_samples);
    xi_u_std = sqrt(xi_u_std / num_samples);
    
    cout << "ξ_f统计: 均值=" << xi_f_mean << ", 标准差=" << xi_f_std << "\n";
    cout << "ξ_f范围: [" << xi_f_min << ", " << xi_f_max << "]\n";
    cout << "ξ_u统计: 均值=" << xi_u_mean << ", 标准差=" << xi_u_std << "\n";
    cout << "ξ_u范围: [" << xi_u_min << ", " << xi_u_max << "]\n";
    
    // 检查是否有样本不满足条件
    int invalid_samples = 0;
    for (const auto& xi_f : xi_f_samples) {
        if (xi_f < lower_bound_val || xi_f > upper_bound_val) {
            invalid_samples++;
        }
    }
    
    if (invalid_samples > 0) {
        cout << "警告: 有" << invalid_samples << "个样本不在指定范围内\n";
    } else {
        cout << "所有样本均在指定范围内\n";
    }
    
    // 准备n的网格
    vector<double> n_grid(n_grid_points);
    for (int i = 0; i < n_grid_points; ++i) {
        n_grid[i] = static_cast<double>(i) / (n_grid_points - 1);
    }
    
    // 存储所有采样的结果
    vector<vector<double>> all_r(num_samples);
    vector<vector<double>> all_n(num_samples);
    vector<vector<double>> all_f(num_samples);
    
    // Step 2: 对于每个采样，计算 n 和 f 随 r 的变化
    cout << "\nStep 2: 对每个采样进行优化计算...\n";
    
    for (int sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
        double xi_f = xi_f_samples[sample_idx];
        double xi_u = xi_u_samples[sample_idx];
        
        // 显示进度
        if (sample_idx % 10 == 0) {
            double progress = 100.0 * (sample_idx + 1) / num_samples;
            cout << "进度: " << sample_idx + 1 << "/" << num_samples 
                     << " (" << progress << "%) - ξ_f=" << xi_f 
                     << ", ξ_u=" << xi_u << "\n";
        }
        
        // 执行优化
        auto [r_values, n_values, f_values] = optimize_for_sample(xi_f, xi_u, n_grid);
        
        // 添加到总结果中
        all_r[sample_idx] = r_values;
        all_n[sample_idx] = n_values;
        all_f[sample_idx] = f_values;
    }
    
    // Step 3: 保存结果到CSV文件
    cout << "\nStep 3: 保存结果...\n";
    save_results(all_r, all_n, all_f, xi_f_samples, xi_u_samples);
    
    // 显示统计信息
    cout << "\n" << string(60, '=') << "\n";
    cout << "计算完成，统计信息:\n";
    cout << string(60, '=') << "\n";
    
    // 展平数据以计算统计信息
    vector<double> all_n_flat;
    vector<double> all_f_flat;
    
    for (const auto& n_vec : all_n) {
        all_n_flat.insert(all_n_flat.end(), n_vec.begin(), n_vec.end());
    }
    
    for (const auto& f_vec : all_f) {
        all_f_flat.insert(all_f_flat.end(), f_vec.begin(), f_vec.end());
    }
    
    cout << "\nn值统计:\n";
    print_stats(all_n_flat, "n");
    
    cout << "\nf值统计:\n";
    print_stats(all_f_flat, "f");
    
    cout << "\n" << string(60, '=') << "\n";
    cout << "所有计算完成！\n";
    cout << string(60, '=') << "\n";
    cout << "结果保存在: " << output_dir << "\n";
    cout << "包含以下文件:\n";
    cout << "  - r_values.csv: r值数据\n";
    cout << "  - n_values.csv: n值数据\n";
    cout << "  - f_values.csv: f值数据\n";
    cout << "  - parameters.csv: 采样参数\n";
    cout << "  - config.csv: 计算配置\n";
    cout << string(60, '=') << "\n";
    
    return 0;
}
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <sstream>
#include <chrono>

using namespace std;

// ==================== 参数设置 ====================
const int NUM_SAMPLES = 100;      // 采样100个ξ_f
const double E = 3.0;            // ΔE = 3
const double U0 = 10.0;          // U0 = 10.0

// ξ_f的分布参数
const double MU_XI_F = 3.0;      // 均值
const double SIGMA_XI_F = 2.5;   // 标准差

// ξ_f的采样范围限制
const double XI_F_MIN = 0.5;     // ξ_f最小值
const double XI_F_MAX = 5.5;     // ξ_f最大值

// 网格设置
const int N_GRID_POINTS = 100;    // n的网格点数 (0到1)
const int R_GRID_POINTS = 1000;   // r的网格点数

// f值范围
const double F_MIN = 0.0;
const double F_MAX = 5.0;
const int F_POINTS = 1000;       // f的采样点数

// 输出路径
const string OUTPUT_DIR = "/home/tyt/project/Single-chain/opt+force/2-state_comparation/results";

// ==================== 辅助函数 ====================
double L_c(double n, double xi_f, double xi_u) {
    return xi_f + n * (xi_u - xi_f);
}

double x_value(double r, double n, double xi_f, double xi_u) {
    double L = L_c(n, xi_f, xi_u);
    if (L <= 0) {
        return NAN;
    }
    return r / L;
}

double F_WLC(double x, double L_c_val) {
    if (x >= 1 || x <= -1 || L_c_val <= 0) {
        return INFINITY;
    }
    double term1 = (M_PI * M_PI) / (2 * L_c_val) * (1 - x * x);
    double term2 = (2 * L_c_val) / (M_PI * (1 - x * x));
    return term1 + term2;
}

double U(double n) {
    return E * n - U0 * cos(2 * M_PI * n);
}

double F_total(double r, double n, double xi_f, double xi_u, double f) {
    double L = L_c(n, xi_f, xi_u);
    double x = x_value(r, n, xi_f, xi_u);
    
    if (isnan(x)) {
        return INFINITY;
    }
    
    return F_WLC(x, L) + U(n) - f * r;
}

struct OptimizationResult {
    double best_r;
    double best_n;
};

OptimizationResult optimize_for_f(double f_value, double xi_f, double xi_u) {
    vector<double> n_values(N_GRID_POINTS);
    vector<double> r_values(R_GRID_POINTS);
    
    for (int i = 0; i < N_GRID_POINTS; ++i) {
        n_values[i] = static_cast<double>(i) / (N_GRID_POINTS - 1);
    }
    
    for (int i = 0; i < R_GRID_POINTS; ++i) {
        r_values[i] = xi_u * static_cast<double>(i) / (R_GRID_POINTS - 1);
    }
    
    double min_energy = INFINITY;
    double best_r = 0.0;
    double best_n = 0.0;
    
    for (double n : n_values) {
        double L = L_c(n, xi_f, xi_u);
        
        for (double r : r_values) {
            if (r > L) {
                continue;
            }
            
            double energy = F_total(r, n, xi_f, xi_u, f_value);
            
            if (energy < min_energy) {
                min_energy = energy;
                best_r = r;
                best_n = n;
            }
        }
    }
    
    return {best_r, best_n};
}

struct SampleResult {
    vector<double> r_results;
    vector<double> n_results;
};

SampleResult process_sample(int sample_idx, double xi_f, double xi_u, const vector<double>& f_values) {
    SampleResult result;
    result.r_results.resize(f_values.size());
    result.n_results.resize(f_values.size());
    
    for (size_t i = 0; i < f_values.size(); ++i) {
        double f = f_values[i];
        auto opt_result = optimize_for_f(f, xi_f, xi_u);
        result.r_results[i] = opt_result.best_r;
        result.n_results[i] = opt_result.best_n;
    }
    
    return result;
}

bool write_csv(const string& filename, const vector<vector<double>>& data) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "无法打开文件: " << filename << endl;
        return false;
    }
    
    file << fixed << setprecision(6);
    
    size_t rows = data[0].size();
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < data.size(); ++j) {
            file << data[j][i];
            if (j < data.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    
    file.close();
    return true;
}

bool write_vector_csv(const string& filename, const vector<double>& data) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "无法打开文件: " << filename << endl;
        return false;
    }
    
    file << fixed << setprecision(6);
    
    for (double value : data) {
        file << value << "\n";
    }
    
    file.close();
    return true;
}

bool write_parameters_csv(const string& filename, 
                         const vector<int>& sample_ids,
                         const vector<double>& xi_f_samples,
                         const vector<double>& xi_u_samples) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "无法打开文件: " << filename << endl;
        return false;
    }
    
    file << "sample_id,xi_f,xi_u\n";
    file << fixed << setprecision(6);
    
    for (size_t i = 0; i < sample_ids.size(); ++i) {
        file << sample_ids[i] << "," << xi_f_samples[i] << "," << xi_u_samples[i] << "\n";
    }
    
    file.close();
    return true;
}

bool write_config_csv(const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "无法打开文件: " << filename << endl;
        return false;
    }
    
    file << "parameter,value,description\n";
    file << "num_samples," << NUM_SAMPLES << ",采样数量\n";
    file << "E," << E << ",ΔE值\n";
    file << "U0," << U0 << ",U0值\n";
    file << "mu_xi_f," << MU_XI_F << ",ξ_f分布的均值\n";
    file << "sigma_xi_f," << SIGMA_XI_F << ",ξ_f分布的标准差\n";
    file << "xi_f_min," << XI_F_MIN << ",ξ_f最小值\n";
    file << "xi_f_max," << XI_F_MAX << ",ξ_f最大值\n";
    file << "f_min," << F_MIN << ",f的最小值\n";
    file << "f_max," << F_MAX << ",f的最大值\n";
    file << "f_points," << F_POINTS << ",f的采样点数\n";
    file << "n_grid_points," << N_GRID_POINTS << ",n的网格点数\n";
    file << "r_grid_points," << R_GRID_POINTS << ",r的网格点数\n";
    
    file.close();
    return true;
}

// 简单的目录创建函数
bool create_directory(const string& path) {
    // 使用系统命令创建目录
    string command = "mkdir -p " + path;
    return system(command.c_str()) == 0;
}

// 采样带有范围限制的ξ_f
pair<vector<double>, vector<double>> sample_xi_f_with_constraint(
    double mu, double sigma, int num_samples, double lower_bound, double upper_bound) {
    
    vector<double> xi_f_samples;
    vector<double> xi_u_samples;
    
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist_xi_f(mu, sigma);
    
    cout << "  正在采样ξ_f，要求范围: [" << lower_bound << ", " << upper_bound << "]" << endl;
    
    int attempts = 0;
    int max_attempts = num_samples * 10;  // 最大尝试次数
    
    while (xi_f_samples.size() < static_cast<size_t>(num_samples) && attempts < max_attempts) {
        double xi_f = dist_xi_f(gen);
        
        if (xi_f >= lower_bound && xi_f <= upper_bound) {
            xi_f_samples.push_back(xi_f);
            xi_u_samples.push_back(10 * xi_f);
        }
        
        attempts++;
    }
    
    // 如果未能采到足够的样本，输出警告
    if (xi_f_samples.size() < static_cast<size_t>(num_samples)) {
        cerr << "警告: 未能采到足够的ξ_f样本 (" 
                  << xi_f_samples.size() << "/" << num_samples << ")" << endl;
    }
    
    // 计算有效采样率
    double success_rate = 100.0 * xi_f_samples.size() / attempts;
    cout << "  采样完成，尝试次数: " << attempts 
              << "，成功: " << xi_f_samples.size() 
              << "，成功率: " << success_rate << "%" << endl;
    
    return {xi_f_samples, xi_u_samples};
}

// ==================== 主函数 ====================
int main() {
    auto start_time = chrono::high_resolution_clock::now();
    
    cout << "=" << string(60, '=') << endl;
    cout << "对比2-state理论与定力系综的结果" << endl;
    cout << "参数: ΔE=" << E << ", U0=" << U0 << endl;
    cout << "采样数: " << NUM_SAMPLES << endl;
    cout << "ξ_f分布: N(" << MU_XI_F << ", " << SIGMA_XI_F << "²)" << endl;
    cout << "ξ_f范围: [" << XI_F_MIN << ", " << XI_F_MAX << "]" << endl;
    cout << "ξ_u = 10 * ξ_f" << endl;
    cout << "=" << string(60, '=') << endl;
    
    // 创建输出目录
    cout << "\n创建输出目录..." << endl;
    if (!create_directory(OUTPUT_DIR)) {
        cerr << "无法创建输出目录: " << OUTPUT_DIR << endl;
        return 1;
    }
    
    // Step 1: 采样ξ_f
    cout << "\nStep 1: 采样" << NUM_SAMPLES << "个ξ_f..." << endl;
    
    // 使用带有范围限制的采样函数
    auto [xi_f_samples, xi_u_samples] = sample_xi_f_with_constraint(
        MU_XI_F, SIGMA_XI_F, NUM_SAMPLES, XI_F_MIN, XI_F_MAX);
    
    // 检查是否采到足够的样本
    if (xi_f_samples.size() < static_cast<size_t>(NUM_SAMPLES)) {
        cerr << "错误: 未能采到足够的ξ_f样本，程序终止" << endl;
        return 1;
    }
    
    // 计算统计信息
    double sum_xi_f = 0.0, sum_xi_u = 0.0;
    double min_xi_f = xi_f_samples[0], max_xi_f = xi_f_samples[0];
    double min_xi_u = xi_u_samples[0], max_xi_u = xi_u_samples[0];
    
    for (size_t i = 0; i < xi_f_samples.size(); ++i) {
        sum_xi_f += xi_f_samples[i];
        sum_xi_u += xi_u_samples[i];
        
        if (xi_f_samples[i] < min_xi_f) min_xi_f = xi_f_samples[i];
        if (xi_f_samples[i] > max_xi_f) max_xi_f = xi_f_samples[i];
        if (xi_u_samples[i] < min_xi_u) min_xi_u = xi_u_samples[i];
        if (xi_u_samples[i] > max_xi_u) max_xi_u = xi_u_samples[i];
    }
    
    double mean_xi_f = sum_xi_f / xi_f_samples.size();
    double mean_xi_u = sum_xi_u / xi_u_samples.size();
    
    double sum_sq_diff_xi_f = 0.0, sum_sq_diff_xi_u = 0.0;
    for (size_t i = 0; i < xi_f_samples.size(); ++i) {
        sum_sq_diff_xi_f += (xi_f_samples[i] - mean_xi_f) * (xi_f_samples[i] - mean_xi_f);
        sum_sq_diff_xi_u += (xi_u_samples[i] - mean_xi_u) * (xi_u_samples[i] - mean_xi_u);
    }
    double std_xi_f = sqrt(sum_sq_diff_xi_f / xi_f_samples.size());
    double std_xi_u = sqrt(sum_sq_diff_xi_u / xi_u_samples.size());
    
    cout << "ξ_f统计: 均值=" << fixed << setprecision(3) 
              << mean_xi_f << ", 标准差=" << std_xi_f << endl;
    cout << "ξ_f范围: [" << min_xi_f << ", " << max_xi_f << "]" << endl;
    cout << "ξ_u统计: 均值=" << mean_xi_u << ", 标准差=" << std_xi_u << endl;
    cout << "ξ_u范围: [" << min_xi_u << ", " << max_xi_u << "]" << endl;
    
    // 检查所有样本是否都在指定范围内
    int out_of_range_count = 0;
    for (size_t i = 0; i < xi_f_samples.size(); ++i) {
        if (xi_f_samples[i] < XI_F_MIN || xi_f_samples[i] > XI_F_MAX) {
            out_of_range_count++;
        }
    }
    
    if (out_of_range_count > 0) {
        cerr << "警告: 有" << out_of_range_count << "个ξ_f样本不在指定范围内" << endl;
    } else {
        cout << "所有ξ_f样本均在指定范围内" << endl;
    }
    
    // 生成f值
    vector<double> f_values(F_POINTS);
    for (int i = 0; i < F_POINTS; ++i) {
        f_values[i] = F_MIN + (F_MAX - F_MIN) * i / (F_POINTS - 1);
    }
    
    // Step 2: 对每个采样进行优化
    cout << "\nStep 2: 对每个采样进行优化..." << endl;
    
    vector<vector<double>> all_r(NUM_SAMPLES);
    vector<vector<double>> all_n(NUM_SAMPLES);
    
    for (int sample_idx = 0; sample_idx < NUM_SAMPLES; ++sample_idx) {
        // 每10个采样输出一次进度
        if (sample_idx % 10 == 0) {
            cout << "进度: " << sample_idx + 1 << "/" << NUM_SAMPLES 
                      << " (" << fixed << setprecision(1) 
                      << 100.0 * (sample_idx + 1) / NUM_SAMPLES << "%)" << endl;
        }
        
        double xi_f = xi_f_samples[sample_idx];
        double xi_u = xi_u_samples[sample_idx];
        
        SampleResult result = process_sample(sample_idx, xi_f, xi_u, f_values);
        
        all_r[sample_idx] = move(result.r_results);
        all_n[sample_idx] = move(result.n_results);
    }
    
    // Step 3: 保存结果
    cout << "\nStep 3: 保存结果到文件..." << endl;
    
    // 保存r值
    if (write_csv(OUTPUT_DIR + "/r_values.csv", all_r)) {
        cout << "  - r_values.csv: 已保存 (" << F_POINTS << "行, " << NUM_SAMPLES << "列)" << endl;
    }
    
    // 保存n值
    if (write_csv(OUTPUT_DIR + "/n_values.csv", all_n)) {
        cout << "  - n_values.csv: 已保存 (" << F_POINTS << "行, " << NUM_SAMPLES << "列)" << endl;
    }
    
    // 保存f值
    if (write_vector_csv(OUTPUT_DIR + "/f_values.csv", f_values)) {
        cout << "  - f_values.csv: 已保存 (" << F_POINTS << "行, 1列)" << endl;
    }
    
    // 保存参数信息
    vector<int> sample_ids(NUM_SAMPLES);
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        sample_ids[i] = i + 1;
    }
    
    if (write_parameters_csv(OUTPUT_DIR + "/parameters.csv", sample_ids, xi_f_samples, xi_u_samples)) {
        cout << "  - parameters.csv: 已保存" << endl;
    }
    
    // 保存配置信息
    if (write_config_csv(OUTPUT_DIR + "/config.csv")) {
        cout << "  - config.csv: 已保存" << endl;
    }
    
    // 计算并保存平均曲线
    vector<double> avg_r(F_POINTS, 0.0);
    vector<double> avg_n(F_POINTS, 0.0);
    
    for (int i = 0; i < F_POINTS; ++i) {
        for (int j = 0; j < NUM_SAMPLES; ++j) {
            avg_r[i] += all_r[j][i];
            avg_n[i] += all_n[j][i];
        }
        avg_r[i] /= NUM_SAMPLES;
        avg_n[i] /= NUM_SAMPLES;
    }
    
    // 保存平均曲线
    ofstream avg_file(OUTPUT_DIR + "/average_curve.csv");
    if (avg_file.is_open()) {
        avg_file << "f,avg_r,avg_n\n";
        avg_file << fixed << setprecision(6);
        for (int i = 0; i < F_POINTS; ++i) {
            avg_file << f_values[i] << "," << avg_r[i] << "," << avg_n[i] << "\n";
        }
        avg_file.close();
        cout << "  - average_curve.csv: 已保存" << endl;
    }
    
    // 计算统计信息
    double min_avg_r = *min_element(avg_r.begin(), avg_r.end());
    double max_avg_r = *max_element(avg_r.begin(), avg_r.end());
    double min_avg_n = *min_element(avg_n.begin(), avg_n.end());
    double max_avg_n = *max_element(avg_n.begin(), avg_n.end());
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);
    
    cout << "\n" << "=" << string(60, '=') << endl;
    cout << "计算完成!" << endl;
    cout << "总耗时: " << duration.count() << " 秒" << endl;
    cout << "结果保存在: " << OUTPUT_DIR << endl;
    cout << "=" << string(60, '=') << endl;
    
    cout << "\n统计信息:" << endl;
    cout << "-" << string(40, '-') << endl;
    
    cout << "f值范围: [" << F_MIN << ", " << F_MAX << "]，共" << F_POINTS << "个点" << endl;
    cout << "平均r值范围: [" << min_avg_r << ", " << max_avg_r << "]" << endl;
    cout << "平均n值范围: [" << min_avg_n << ", " << max_avg_n << "]" << endl;
    
    return 0;
}
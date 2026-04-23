// 系统设置：N个折叠domain串联起来的链
// 结构异质性：每个domain打开的能量惩罚服从高斯分布
// 本程序用于优化自由能函数


#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <numeric>
#include <limits>
#include <string>
#include <filesystem>
#include <sstream>

using namespace std;
namespace fs = filesystem;

// ==================== 参数设置 ====================
const int N = 5;              // domain数量
const int Number = 100;       // 采样链的数量
const double xi_f = 10.0;     // 折叠状态长度
const double k = 2.0;         // 比例系数
const double xi_u = k * xi_f; // 展开状态长度
const double U0 = 20.0;       // U0值
const double DeltaE_mean = 10.0;  // DeltaE高斯分布的均值
const double DeltaE_std = 5.0;    // DeltaE高斯分布的标准差

// 采样范围限制（避免使用lower_bound/upper_bound，与STL函数冲突）
const double min_val = 1.0;
const double max_val = 30.0;

// 网格设置
const double f_max = 10.0;    // 最大力值
const int f_grid = 200;       // f的网格点数
const int r_grid = 1000;
const int n_grid = 200;

// ==================== 保存路径 ====================
const string SAVE_DIR = "/home/tyt/project/Single-chain/opt+force/Rand_energy/results";

// ==================== 辅助函数 ====================
vector<double> linspace(double start, double stop, int num) {
    vector<double> result(num);
    double step = (stop - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }
    return result;
}

vector<double> sort_vector(const vector<double>& v) {
    vector<double> sorted = v;
    sort(sorted.begin(), sorted.end());
    return sorted;
}

// ==================== 生成随机数 ====================
class RandomGenerator {
private:
    mt19937 gen;
    normal_distribution<double> normal_dist;
    uniform_real_distribution<double> uniform_dist;
    
public:
    RandomGenerator(double mean, double stddev, double seed = 42)
        : gen(seed), normal_dist(mean, stddev), uniform_dist(0.0, 1.0) {}
    
    double get_normal() {
        return normal_dist(gen);
    }
    
    double get_uniform() {
        return uniform_dist(gen);
    }
};

// ==================== Step 1: 生成Energy.csv ====================
void generate_energy(const string& save_dir) {
    cout << "Step 1: 生成Energy.csv..." << endl;
    cout << "采样范围: [" << min_val << ", " << max_val << "]" << endl;
    
    // 创建目录
    fs::create_directories(save_dir);
    
    // 打开文件
    string energy_path = save_dir + "/Energy.csv";
    ofstream file(energy_path);
    if (!file.is_open()) {
        cerr << "无法打开文件: " << energy_path << endl;
        return;
    }
    
    // 生成随机数
    RandomGenerator rng(DeltaE_mean, DeltaE_std);
    
    // 写入数据
    for (int i = 0; i < Number; ++i) {
        file << i;
        
        vector<double> delta_es(N);
        bool all_in_range = false;
        int attempts = 0;
        const int max_attempts = 1000;
        
        // 尝试生成在范围内的随机数
        while (!all_in_range && attempts < max_attempts) {
            ++attempts;
            all_in_range = true;
            
            for (int j = 0; j < N; ++j) {
                delta_es[j] = rng.get_normal();
                if (delta_es[j] < min_val || delta_es[j] > max_val) {
                    all_in_range = false;
                }
            }
        }
        
        // 如果超过最大尝试次数，使用截断方法
        if (!all_in_range) {
            cout << "警告: 链 " << i << " 在" << max_attempts 
                      << "次尝试后仍未生成完全在范围内的值，使用截断方法" << endl;
            
            // 重新生成并截断
            for (int j = 0; j < N; ++j) {
                double val = rng.get_normal();
                if (val < min_val) val = min_val;
                if (val > max_val) val = max_val;
                delta_es[j] = val;
            }
        }
        
        // 写入DeltaE值
        for (double val : delta_es) {
            file << "," << val;
        }
        file << "\n";
    }
    
    file.close();
    cout << "已生成Energy.csv，保存到: " << energy_path << endl;
    cout << "包含" << Number << "行数据" << endl;
}

// ==================== Step 2: 读取Energy.csv并计算U_int函数 ====================
class UIntInterpolator {
private:
    vector<double> n_values;
    vector<double> U_int_values;
    
public:
    UIntInterpolator(const vector<double>& sorted_delta_es) {
        // 创建n值数组: 0, 1, 2, ..., N
        n_values.resize(N + 1);
        for (int i = 0; i <= N; ++i) {
            n_values[i] = i;
        }
        
        // 计算整数n处的U_int值
        U_int_values.resize(N + 1);
        U_int_values[0] = 0.0;
        
        double sum = 0.0;
        for (int i = 1; i <= N; ++i) {
            sum += sorted_delta_es[i - 1];
            U_int_values[i] = sum;
        }
    }
    
    double operator()(double n) const {
        // 线性插值
        if (n <= 0.0) return U_int_values[0];
        if (n >= static_cast<double>(N)) return U_int_values[N];
        
        int i = static_cast<int>(floor(n));
        double t = n - i;
        
        return (1.0 - t) * U_int_values[i] + t * U_int_values[i + 1];
    }
};

vector<UIntInterpolator> read_energy_and_create_interpolators(const string& save_dir) {
    cout << "\nStep 2: 读取Energy.csv并计算U_int函数..." << endl;
    
    string energy_path = save_dir + "/Energy.csv";
    ifstream file(energy_path);
    if (!file.is_open()) {
        cerr << "无法打开文件: " << energy_path << endl;
        return {};
    }
    
    vector<UIntInterpolator> interpolators;
    string line;
    
    // 跳过第一行（如果有表头）
    getline(file, line);
    
    while (getline(file, line)) {
        vector<double> delta_es;
        stringstream ss(line);
        string token;
        
        // 跳过索引
        getline(ss, token, ',');
        
        // 读取DeltaE值
        while (getline(ss, token, ',')) {
            delta_es.push_back(stod(token));
        }
        
        // 对DeltaE排序
        vector<double> sorted_delta_es = sort_vector(delta_es);
        
        // 创建插值器
        interpolators.emplace_back(sorted_delta_es);
    }
    
    file.close();
    cout << "已计算" << interpolators.size() << "条链的U_int插值函数" << endl;
    return interpolators;
}

// ==================== 计算WLC自由能函数 ====================
double F_WLC(double x, double L_c) {
    // F_WLC(x, L_c) = (π²/(2L_c)) * (1 - x²) + (2L_c)/(π(1-x²))
    double term1 = (M_PI * M_PI / (2.0 * L_c)) * (1.0 - x * x);
    double term2 = (2.0 * L_c) / (M_PI * (1.0 - x * x));
    return term1 + term2;
}

// ==================== 计算单链自由能 ====================
double F_c(double r, double n, double f, const UIntInterpolator& U_int_func) {
    // 计算轮廓长度
    double L_c_val = N * xi_f + n * (xi_u - xi_f);
    
    // 避免除零错误
    if (L_c_val <= 0.0) {
        return numeric_limits<double>::infinity();
    }
    
    // 计算端到端因子
    double x_val = r / L_c_val;
    
    // 避免x>=1导致WLC公式发散
    if (x_val >= 1.0 || x_val <= -1.0) {
        return numeric_limits<double>::infinity();
    }
    
    // 计算WLC自由能
    double F_WLC_val = F_WLC(x_val, L_c_val);
    
    // 计算U(n) = U_int(n) - U0 * cos(2πn)
    double U_int_val = U_int_func(n);
    double U_val = U_int_val - U0 * cos(2.0 * M_PI * n);
    
    // 计算总自由能
    return F_WLC_val + U_val - f * r;
}

// ==================== Step 3: 计算最优(r, n)并保存结果 ====================
void calculate_optimal_r_n(const vector<UIntInterpolator>& interpolators, const string& save_dir) {
    cout << "\nStep 3: 计算最优(r, n)并保存结果..." << endl;
    
    // 生成f值
    vector<double> f_values = linspace(0.0, f_max, f_grid);
    
    // 保存f_values到文件
    string f_path = save_dir + "/f_values.csv";
    ofstream f_file(f_path);
    for (double f : f_values) {
        f_file << f << "\n";
    }
    f_file.close();
    cout << "已保存f_values到: " << f_path << ", 数量: " << f_values.size() << endl;
    
    // 生成r值
    double r_max = N * xi_u;
    vector<double> r_values = linspace(0.0, r_max, r_grid);
    
    // 生成n值
    vector<double> n_values = linspace(0.0, N, n_grid);
    
    // 初始化存储数组
    vector<vector<double>> r_opt_matrix(Number, vector<double>(f_grid));
    vector<vector<double>> n_opt_matrix(Number, vector<double>(f_grid));
    
    // 对每条链进行计算
    for (int chain_idx = 0; chain_idx < Number; ++chain_idx) {
        const UIntInterpolator& U_int_func = interpolators[chain_idx];
        
        // 对每个f值进行计算
        for (int f_idx = 0; f_idx < f_grid; ++f_idx) {
            double f = f_values[f_idx];
            double min_F = numeric_limits<double>::infinity();
            double r_opt = 0.0;
            double n_opt = 0.0;
            
            // 扫描r和n，寻找最小自由能
            for (double r : r_values) {
                for (double n : n_values) {
                    double F_val = F_c(r, n, f, U_int_func);
                    
                    // 更新最小值
                    if (F_val < min_F) {
                        min_F = F_val;
                        r_opt = r;
                        n_opt = n;
                    }
                }
            }
            
            // 存储最优值
            r_opt_matrix[chain_idx][f_idx] = r_opt;
            n_opt_matrix[chain_idx][f_idx] = n_opt;
        }
        
        // 打印进度
        if ((chain_idx + 1) % 10 == 0) {
            cout << "  已完成" << (chain_idx + 1) << "/" << Number << "条链的计算" << endl;
        }
    }
    
    // 保存r_opt到文件
    string r_path = save_dir + "/r_values.csv";
    ofstream r_file(r_path);
    for (int i = 0; i < Number; ++i) {
        for (int j = 0; j < f_grid; ++j) {
            r_file << r_opt_matrix[i][j];
            if (j < f_grid - 1) r_file << ",";
        }
        r_file << "\n";
    }
    r_file.close();
    
    // 保存n_opt到文件
    string n_path = save_dir + "/n_values.csv";
    ofstream n_file(n_path);
    for (int i = 0; i < Number; ++i) {
        for (int j = 0; j < f_grid; ++j) {
            n_file << n_opt_matrix[i][j];
            if (j < f_grid - 1) n_file << ",";
        }
        n_file << "\n";
    }
    n_file.close();
    
    cout << "已保存结果到:" << endl;
    cout << "  " << r_path << ", 形状: " << Number << "行×" << f_grid << "列" << endl;
    cout << "  " << n_path << ", 形状: " << Number << "行×" << f_grid << "列" << endl;
    cout << "数据格式:" << endl;
    cout << "  f_values.csv: " << f_grid << "行×1列" << endl;
    cout << "  r_values.csv: " << Number << "行×" << f_grid << "列，每行对应一条链，每列对应一个f值" << endl;
    cout << "  n_values.csv: " << Number << "行×" << f_grid << "列，每行对应一条链，每列对应一个f值" << endl;
}

// ==================== 主程序 ====================
int main() {
    cout << "============================================================" << endl;
    cout << "结构异质性的影响分析程序 (C++版本)" << endl;
    cout << "参数设置:" << endl;
    cout << "  N = " << N << ", Number = " << Number << endl;
    cout << "  xi_f = " << xi_f << ", xi_u = " << xi_u << ", U0 = " << U0 << endl;
    cout << "  DeltaE_mean = " << DeltaE_mean << ", DeltaE_std = " << DeltaE_std << endl;
    cout << "  采样范围: [" << min_val << ", " << max_val << "]" << endl;
    cout << "  结果保存到: " << SAVE_DIR << endl;
    cout << "============================================================" << endl;
    
    // Step 1: 生成Energy.csv
    generate_energy(SAVE_DIR);
    
    // Step 2: 读取Energy.csv并计算U_int函数
    auto interpolators = read_energy_and_create_interpolators(SAVE_DIR);
    if (interpolators.empty()) {
        cerr << "错误: 无法创建插值函数" << endl;
        return 1;
    }
    
    // Step 3: 计算最优(r, n)并保存结果
    calculate_optimal_r_n(interpolators, SAVE_DIR);
    
    cout << "\n============================================================" << endl;
    cout << "程序执行完成!" << endl;
    cout << "============================================================" << endl;
    
    return 0;
}
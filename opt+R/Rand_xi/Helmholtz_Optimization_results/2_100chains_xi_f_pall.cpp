/**
 * 多域链本构曲线计算 - C++/OpenMP并行版本（使用 using namespace std）
 * 编译：g++ -std=c++17 -O3 -march=native -fopenmp -o chain_sim main.cpp
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <limits>
#include <chrono>
#include <omp.h>

using namespace std;

// ==================== 参数常量 ====================
const int N_samples = 100;          // 样本数（链数）
const int Num = 2;                  // 每条链的domain数
const double mu = 20.0;             // xi_f 高斯分布均值
const double sigma = 7.0;           // xi_f 标准差
const double delta = 5.0;           // 截断范围 [mu-delta, mu+delta]

const double alpha = 2.0;           // xi_u = alpha * xi_f
const double E0 = 5.0;              // 能量的最小值
const double Ek = 2.0;           // 能量与xi_f的系数
const int r_grids = 500;            // 每条链的拉伸长度网格点数

const int init_points = 30;         // 初始粗网格点数
const int refine_levels = 4;        // 细化层数
const int refine_points = 20;       // 每层细化点数
const double tol = 1e-6;            // 细化收敛容差

const double EPS_BOUND = 1e-12;      // 边界零长度容差
const double GOLDEN_RATIO = 1.6180339887498948482;
const double GOLDEN_SECTION_TOL = 1e-8;
const int MAX_GOLDEN_ITER = 100;

const string output_dir = "/home/tyt/project/Single-chain/opt+R/Rand_xi/Helmholtz_Optimization_results/2-domain_xi_f_pall_results";   // 请确保目录存在

// ==================== 辅助函数 ====================
vector<double> linspace(double start, double end, int num) {
    vector<double> res(num);
    if (num <= 1) {
        res[0] = start;
        return res;
    }
    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        res[i] = start + i * step;
    }
    return res;
}

double contour_length_Lci(double n_i, double xi_fi) {
    return xi_fi + n_i * (alpha - 1.0) * xi_fi;
}

double energy_term_U(double n_i, double DeltaE) {
    return DeltaE * n_i - DeltaE * cos(2.0 * M_PI * n_i);
}

double WLC_free_energy(double x, double Lc) {
    if (x >= 0.999999) {
        return 1e100;
    } else {
        return 0.25 * Lc * (x * x * (3.0 - 2.0 * x) / (1.0 - x));
    }
}

double single_domain_free_energy(double r_i, double n_i, double DeltaE, double xi_fi) {
    double Lc = contour_length_Lci(n_i, xi_fi);
    if (r_i < 0.0 || r_i >= Lc) return 1e300;
    double x = r_i / Lc;
    double F_wlc = WLC_free_energy(x, Lc);
    double Ui = energy_term_U(n_i, DeltaE);
    return F_wlc + Ui;
}

double free_energy_2_domain(double r, double r1, double n1, double n2,
                            double DeltaE1, double DeltaE2,
                            double xi_f1, double xi_f2) {
    double energy1 = single_domain_free_energy(r1, n1, DeltaE1, xi_f1);
    double r2 = r - r1;
    double energy2 = single_domain_free_energy(r2, n2, DeltaE2, xi_f2);
    double total = energy1 + energy2;
    return isfinite(total) ? total : 1e300;
}

struct GoldenResult {
    double x;
    double fval;
    bool success;
};


// 模板版本：可接受任何可调用对象
template<typename Func>
GoldenResult golden_section_search(Func func, double a, double b, double tol = GOLDEN_SECTION_TOL) {
    double c = b - (b - a) / GOLDEN_RATIO;
    double d = a + (b - a) / GOLDEN_RATIO;
    double fc = func(c);
    double fd = func(d);

    for (int iter = 0; iter < MAX_GOLDEN_ITER; ++iter) {
        if (fc < fd) {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) / GOLDEN_RATIO;
            fc = func(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) / GOLDEN_RATIO;
            fd = func(d);
        }
        if (abs(b - a) < tol) break;
    }
    double xmin = (a + b) / 2.0;
    double fmin = func(xmin);
    return {xmin, fmin, true};
}

struct OptResult {
    double r1;
    double n1;
    double n2;
    double free_energy;
};

OptResult optimize_single_point(double r, double DeltaE1, double DeltaE2,
                                double xi_f1, double xi_f2) {
    double n1_min = 0.0, n1_max = 1.0;
    double n2_min = 0.0, n2_max = 1.0;
    double best_r1 = numeric_limits<double>::quiet_NaN();
    double best_n1 = numeric_limits<double>::quiet_NaN();
    double best_n2 = numeric_limits<double>::quiet_NaN();
    double best_F = numeric_limits<double>::infinity();

    for (int level = 0; level <= refine_levels; ++level) {
        int N = (level == 0) ? init_points : refine_points;
        vector<double> n1_grid = linspace(n1_min, n1_max, N);
        vector<double> n2_grid = linspace(n2_min, n2_max, N);

        double level_best_F = numeric_limits<double>::infinity();
        double level_best_r1 = 0.0, level_best_n1 = 0.0, level_best_n2 = 0.0;

        for (double n1 : n1_grid) {
            double Lc1 = contour_length_Lci(n1, xi_f1);
            for (double n2 : n2_grid) {
                double Lc2 = contour_length_Lci(n2, xi_f2);
                double r1_lower = max(0.0, r - min(r, Lc2));
                double r1_upper = min(Lc1, r);
                if (r1_lower > r1_upper) continue;

                if (r1_upper - r1_lower < EPS_BOUND) {
                    double r1_mid = (r1_lower + r1_upper) * 0.5;
                    double F_val = free_energy_2_domain(r, r1_mid, n1, n2, DeltaE1, DeltaE2, xi_f1, xi_f2);
                    if (F_val < level_best_F) {
                        level_best_F = F_val;
                        level_best_r1 = r1_mid;
                        level_best_n1 = n1;
                        level_best_n2 = n2;
                    }
                    continue;
                }

                auto func = [&](double r1) {
                    return free_energy_2_domain(r, r1, n1, n2, DeltaE1, DeltaE2, xi_f1, xi_f2);
                };
                GoldenResult res = golden_section_search(func, r1_lower, r1_upper);
                if (res.success && res.fval < level_best_F) {
                    level_best_F = res.fval;
                    level_best_r1 = res.x;
                    level_best_n1 = n1;
                    level_best_n2 = n2;
                }
            }
        }

        if (!isfinite(level_best_F)) {
            if (!isfinite(best_F)) {
                return {nan(""), nan(""), nan(""), nan("")};
            } else {
                break;
            }
        }

        if (level_best_F < best_F) {
            best_F = level_best_F;
            best_r1 = level_best_r1;
            best_n1 = level_best_n1;
            best_n2 = level_best_n2;
        }

        if (level == refine_levels) break;

        auto find_idx = [](const vector<double>& grid, double val) {
            auto it = min_element(grid.begin(), grid.end(),
                [val](double a, double b) { return abs(a - val) < abs(b - val); });
            return static_cast<int>(it - grid.begin());
        };
        int idx_n1 = find_idx(n1_grid, best_n1);
        int idx_n2 = find_idx(n2_grid, best_n2);
        int N1 = static_cast<int>(n1_grid.size());
        int N2 = static_cast<int>(n2_grid.size());
        double left_n1 = n1_grid[max(0, idx_n1 - 1)];
        double right_n1 = n1_grid[min(N1 - 1, idx_n1 + 1)];
        double left_n2 = n2_grid[max(0, idx_n2 - 1)];
        double right_n2 = n2_grid[min(N2 - 1, idx_n2 + 1)];
        n1_min = left_n1; n1_max = right_n1;
        n2_min = left_n2; n2_max = right_n2;

        if ((n1_max - n1_min < tol) && (n2_max - n2_min < tol)) break;
    }

    return {best_r1, best_n1, best_n2, best_F};
}

double truncated_normal_sample(double mu, double sigma, double low, double high, mt19937& gen) {
    normal_distribution<double> norm(mu, sigma);
    double x;
    do {
        x = norm(gen);
    } while (x < low || x > high);
    return x;
}

void write_column_csv(const string& filename,
                      const vector<vector<double>>& data_columns,
                      int num_rows) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "无法打开文件 " << filename << endl;
        return;
    }
    file << setprecision(15);
    for (int i = 0; i < static_cast<int>(data_columns.size()); ++i) {
        file << "chain_" << i+1;
        if (i != static_cast<int>(data_columns.size())-1) file << ",";
    }
    file << "\n";
    for (int row = 0; row < num_rows; ++row) {
        for (int col = 0; col < static_cast<int>(data_columns.size()); ++col) {
            file << data_columns[col][row];
            if (col != static_cast<int>(data_columns.size())-1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

int main() {
    // system(("mkdir -p " + output_dir).c_str());

    auto start_time = chrono::steady_clock::now();

    // ---------- Step 1: 采样 xi_f ----------
    random_device rd;
    mt19937 gen(rd());
    double low = mu - delta;
    double high = mu + delta;
    vector<double> xi_f1_samples(N_samples);
    vector<double> xi_f2_samples(N_samples);
    for (int i = 0; i < N_samples; ++i) {
        xi_f1_samples[i] = truncated_normal_sample(mu, sigma, low, high, gen);
        xi_f2_samples[i] = truncated_normal_sample(mu, sigma, low, high, gen);
    }

    ofstream xi_f_file(output_dir + "/xi_f.csv");
    xi_f_file << "group,xi_f1,xi_f2\n";
    for (int i = 0; i < N_samples; ++i) {
        xi_f_file << i+1 << "," << xi_f1_samples[i] << "," << xi_f2_samples[i] << "\n";
    }
    xi_f_file.close();
    cout << "Step 1 完成：xi_f 采样已保存" << endl;

    // ---------- Step 2: 计算 ΔE ----------
    vector<double> DeltaE1_samples(N_samples);
    vector<double> DeltaE2_samples(N_samples);
    for (int i = 0; i < N_samples; ++i) {
        DeltaE1_samples[i] = E0 + Ek * (xi_f1_samples[i] - mu + delta);
        DeltaE2_samples[i] = E0 + Ek * (xi_f2_samples[i] - mu + delta);
    }

    ofstream energy_file(output_dir + "/DeltaE.csv");
    energy_file << "group,DeltaE1,DeltaE2\n";
    for (int i = 0; i < N_samples; ++i) {
        energy_file << i+1 << "," << DeltaE1_samples[i] << "," << DeltaE2_samples[i] << "\n";
    }
    energy_file.close();
    cout << "Step 2 完成：ΔE 已保存" << endl;

    // ---------- Step 3 & 4: 并行计算每条链 ----------
    vector<vector<double>> all_r1(N_samples);
    vector<vector<double>> all_r2(N_samples);
    vector<vector<double>> all_n1(N_samples);
    vector<vector<double>> all_n2(N_samples);
    vector<vector<double>> all_r_vals(N_samples);

    #pragma omp parallel for schedule(dynamic) num_threads(8)
    for (int i = 0; i < N_samples; ++i) {
        double xf1 = xi_f1_samples[i];
        double xf2 = xi_f2_samples[i];
        double dE1 = DeltaE1_samples[i];
        double dE2 = DeltaE2_samples[i];

        double Lc1_max = contour_length_Lci(1.0, xf1);
        double Lc2_max = contour_length_Lci(1.0, xf2);
        double total_max = Lc1_max + Lc2_max;
        double r_max = 0.95 * total_max;

        vector<double> r_vals = linspace(0.0, r_max, r_grids);
        vector<double> cur_r1(r_grids), cur_r2(r_grids), cur_n1(r_grids), cur_n2(r_grids);

        for (int j = 0; j < r_grids; ++j) {
            double r = r_vals[j];
            OptResult opt = optimize_single_point(r, dE1, dE2, xf1, xf2);
            cur_r1[j] = opt.r1;
            cur_n1[j] = opt.n1;
            cur_n2[j] = opt.n2;
            cur_r2[j] = r - opt.r1;
        }

        all_r1[i] = move(cur_r1);
        all_r2[i] = move(cur_r2);
        all_n1[i] = move(cur_n1);
        all_n2[i] = move(cur_n2);
        all_r_vals[i] = move(r_vals);

        #pragma omp critical
        {
            cout << "已完成第 " << i+1 << "/" << N_samples << " 条链: xi_f1=" << xf1
                 << ", xi_f2=" << xf2 << ", r_max=" << r_max << endl;
        }
    }

    // ---------- Step 5: 保存结果 ----------
    write_column_csv(output_dir + "/r_vals.csv", all_r_vals, r_grids);
    write_column_csv(output_dir + "/r1_values.csv", all_r1, r_grids);
    write_column_csv(output_dir + "/r2_values.csv", all_r2, r_grids);
    write_column_csv(output_dir + "/n1_values.csv", all_n1, r_grids);
    write_column_csv(output_dir + "/n2_values.csv", all_n2, r_grids);

    auto end_time = chrono::steady_clock::now();
    double elapsed = chrono::duration<double>(end_time - start_time).count();
    cout << "\n所有结果已保存至目录：" << output_dir << endl;
    cout << "计算耗时: " << elapsed << " 秒" << endl;

    return 0;
}
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>

// 数据点结构体
struct DataPoint {
    double R;
    double f_red;   // 红线值
};

// 区域结构体
struct Region {
    std::string name;    // "a", "a'", "b", "b'", etc.
    double area;
    double black_line_value; // 该区域对应的黑线值
    double start_R;      // 区域起始R坐标（第一个交点）
    double end_R;        // 区域结束R坐标（第二个交点）
};

// Maxwell构造结构体
struct MaxwellConstruction {
    char name;                  // 'a', 'b', 'c'...
    double black_line_value;    // 黑线值（来自蓝线跳变）
    double intersection1_R;     // 第一个交点（蓝线跳变前的r）
    double intersection2_R;     // 第二个交点（蓝线跳变后的r） 
    double critical_R;          // 红线关键点r_c（f跳变点）
    std::vector<Region> regions; // 该构造包含的区域
};

class MaxwellConstructionIntegrator {
private:
    std::vector<DataPoint> red_data;
    std::vector<std::pair<double, double>> blue_data;

public:
    // 构造函数
    MaxwellConstructionIntegrator(const std::vector<DataPoint>& red_data_input,
                                 const std::vector<std::pair<double, double>>& blue_data_input)
        : red_data(red_data_input), blue_data(blue_data_input) {}

    // 读取CSV文件
    static bool readDataFromCSV(const std::string& red_file, const std::string& blue_file,
                               std::vector<DataPoint>& red_data_out,
                               std::vector<std::pair<double, double>>& blue_data_out) {
        red_data_out = readRedLineData(red_file);
        blue_data_out = readBlueLineData(blue_file);
        
        return !red_data_out.empty() && !blue_data_out.empty();
    }

private:
    // 读取红线数据
    static std::vector<DataPoint> readRedLineData(const std::string& filename) {
        std::vector<DataPoint> red_data;
        std::ifstream file(filename);
        std::string line;
        
        if (!file.is_open()) {
            std::cerr << "错误：无法打开红线文件 " << filename << std::endl;
            return red_data;
        }
        
        // 跳过第一行标题
        std::getline(file, line);
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row;
            
            // 解析CSV行
            while (std::getline(ss, cell, ',')) {
                row.push_back(cell);
            }
            
            if (row.size() >= 4) {
                try {
                    DataPoint point;
                    point.R = std::stod(row[0]);  // 第一列 R
                    point.f_red = std::stod(row[3]);  // 第四列 f
                    red_data.push_back(point);
                } catch (const std::exception& e) {
                    std::cerr << "警告：跳过无法解析的行: " << line << std::endl;
                }
            }
        }
        
        // 按R值排序
        std::sort(red_data.begin(), red_data.end(),
                 [](const DataPoint& a, const DataPoint& b) { return a.R < b.R; });
        
        std::cout << "从 " << filename << " 读取了 " << red_data.size() << " 个红线数据点" << std::endl;
        return red_data;
    }
    
    // 读取蓝线数据
    static std::vector<std::pair<double, double>> readBlueLineData(const std::string& filename) {
        std::vector<std::pair<double, double>> blue_data;
        std::ifstream file(filename);
        std::string line;
        
        if (!file.is_open()) {
            std::cerr << "错误：无法打开蓝线文件 " << filename << std::endl;
            return blue_data;
        }
        
        // 跳过第一行标题
        std::getline(file, line);
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row;
            
            // 解析CSV行
            while (std::getline(ss, cell, ',')) {
                row.push_back(cell);
            }
            
            if (row.size() >= 2) {
                try {
                    double f = std::stod(row[0]);  // 第一列 f
                    double R = std::stod(row[1]);  // 第二列 R
                    blue_data.push_back({R, f});
                } catch (const std::exception& e) {
                    std::cerr << "警告：跳过无法解析的行: " << line << std::endl;
                }
            }
        }
        
        // 按R值排序
        std::sort(blue_data.begin(), blue_data.end(),
                 [](const auto& a, const auto& b) { return a.first < b.first; });
        
        std::cout << "从 " << filename << " 读取了 " << blue_data.size() << " 个蓝线数据点" << std::endl;
        return blue_data;
    }

public:
    // 检测所有Maxwell构造
    std::vector<MaxwellConstruction> detectMaxwellConstructions(double jump_threshold_ratio = 5.0) {
        std::vector<MaxwellConstruction> constructions;
        
        if (blue_data.size() < 2 || red_data.size() < 2) {
            std::cerr << "错误：数据点不足" << std::endl;
            return constructions;
        }
        
        // 计算蓝线平均步长
        double blue_avg_step = calculateAverageStep(blue_data);
        double blue_jump_threshold = blue_avg_step * jump_threshold_ratio;
        
        // 计算红线平均步长（用于检测红线跳变）
        double red_avg_step = calculateAverageStepRed();
        double red_jump_threshold = red_avg_step * jump_threshold_ratio;
        
        std::cout << "蓝线平均步长: " << blue_avg_step << ", 跳变阈值: " << blue_jump_threshold << std::endl;
        std::cout << "红线平均步长: " << red_avg_step << ", 跳变阈值: " << red_jump_threshold << std::endl;
        
        // 检测蓝线跳变区域（确定黑线值和交点）
        std::vector<MaxwellConstruction> blue_constructions = detectBlueJumpRegions(blue_jump_threshold);
        
        // 检测红线关键点r_c
        std::vector<double> red_critical_points = detectRedCriticalPoints(red_jump_threshold);
        
        if (blue_constructions.size() != 5 || red_critical_points.size() != 5) {
            std::cout << "警告：期望5个构造和5个关键点，但检测到 " 
                      << blue_constructions.size() << " 个构造和 "
                      << red_critical_points.size() << " 个关键点" << std::endl;
        }
        
        // 合并蓝线构造和红线关键点
        for (size_t i = 0; i < std::min(blue_constructions.size(), red_critical_points.size()); ++i) {
            MaxwellConstruction construction = blue_constructions[i];
            construction.name = 'a' + i;
            construction.critical_R = red_critical_points[i];
            
            std::cout << "构造 " << construction.name << ":" << std::endl;
            std::cout << "  交点1 (蓝线跳变前): R=" << construction.intersection1_R << std::endl;
            std::cout << "  交点2 (蓝线跳变后): R=" << construction.intersection2_R << std::endl;
            std::cout << "  黑线值: f=" << construction.black_line_value << std::endl;
            std::cout << "  红线关键点 r_c: R=" << construction.critical_R << std::endl;
            
            // 验证关键点在两个交点之间
            if (construction.critical_R < construction.intersection1_R || 
                construction.critical_R > construction.intersection2_R) {
                std::cout << "  警告：关键点r_c不在交点范围内！自动调整..." << std::endl;
                // 自动调整到两个交点的中点
                construction.critical_R = (construction.intersection1_R + construction.intersection2_R) / 2.0;
            }
            
            // 计算区域面积
            construction.regions = calculateRegionsForConstruction(construction);
            constructions.push_back(construction);
        }
        
        return constructions;
    }

private:
    // 计算平均步长（蓝线）
    double calculateAverageStep(const std::vector<std::pair<double, double>>& data) {
        double total_step = 0.0;
        int step_count = 0;
        for (size_t i = 1; i < data.size(); ++i) {
            double step = data[i].first - data[i-1].first;
            if (step > 0) {
                total_step += step;
                step_count++;
            }
        }
        return step_count > 0 ? total_step / step_count : 0.0;
    }
    
    // 计算平均步长（红线）
    double calculateAverageStepRed() {
        double total_step = 0.0;
        int step_count = 0;
        for (size_t i = 1; i < red_data.size(); ++i) {
            double step = red_data[i].R - red_data[i-1].R;
            if (step > 0) {
                total_step += step;
                step_count++;
            }
        }
        return step_count > 0 ? total_step / step_count : 0.0;
    }
    
    // 检测蓝线跳变区域并确定交点
    std::vector<MaxwellConstruction> detectBlueJumpRegions(double threshold) {
        std::vector<MaxwellConstruction> constructions;
        
        for (size_t i = 1; i < blue_data.size(); ++i) {
            double R_prev = blue_data[i-1].first;
            double R_curr = blue_data[i].first;
            double step = R_curr - R_prev;
            
            if (step > threshold) {
                MaxwellConstruction construction;
                // 交点1：跳变前的r值
                construction.intersection1_R = R_prev;
                // 交点2：跳变后的r值  
                construction.intersection2_R = R_curr;
                // 黑线值：跳变前的f值
                construction.black_line_value = blue_data[i-1].second;
                
                constructions.push_back(construction);
                
                std::cout << "发现蓝线跳变区域 " << constructions.size() << ":" << std::endl;
                std::cout << "  交点1 (跳变前): R=" << construction.intersection1_R << std::endl;
                std::cout << "  交点2 (跳变后): R=" << construction.intersection2_R << std::endl;
                std::cout << "  黑线值: f=" << construction.black_line_value << std::endl;
                std::cout << "  跳变步长: " << step << std::endl;
            }
        }
        
        return constructions;
    }
    
    // 检测红线关键点r_c
    std::vector<double> detectRedCriticalPoints(double threshold) {
        std::vector<double> critical_points;
        
        for (size_t i = 1; i < red_data.size(); ++i) {
            double step_R = red_data[i].R - red_data[i-1].R;
            double step_f = std::abs(red_data[i].f_red - red_data[i-1].f_red);
            
            // 如果R步长正常但f值变化很大，认为是红线跳变
            if (step_R <= threshold * 2 && step_f > threshold * 2) {
                // 取两个点的中点作为关键点位置
                double critical_R = (red_data[i-1].R + red_data[i].R) / 2.0;
                critical_points.push_back(critical_R);
                std::cout << "发现红线关键点 " << critical_points.size() << ": R=" << critical_R
                          << " (f从 " << red_data[i-1].f_red << " 到 " << red_data[i].f_red 
                          << ", Δf=" << step_f << ")" << std::endl;
            }
        }
        
        return critical_points;
    }
    
    // 计算单个Maxwell构造的区域面积
    std::vector<Region> calculateRegionsForConstruction(const MaxwellConstruction& construction) {
        std::vector<Region> regions;
        
        // 区域a：从交点1到关键点r_c，积分 (f_red - f_black)
        double area_a = integrateArea(construction.intersection1_R, construction.critical_R, 
                                     construction.black_line_value, false);
        
        Region region_a;
        region_a.name = std::string(1, construction.name);
        region_a.area = std::abs(area_a);
        region_a.black_line_value = construction.black_line_value;
        region_a.start_R = construction.intersection1_R;
        region_a.end_R = construction.critical_R;
        regions.push_back(region_a);
        
        std::cout << "  区域 " << region_a.name << ": R=[" << construction.intersection1_R 
                  << ", " << construction.critical_R << "], 面积=" << region_a.area 
                  << " (积分 f_red - f_black)" << std::endl;
        
        // 区域a'：从关键点r_c到交点2，积分 (f_black - f_red)
        double area_a_prime = integrateArea(construction.critical_R, construction.intersection2_R, 
                                          construction.black_line_value, true);
        
        Region region_a_prime;
        region_a_prime.name = std::string(1, construction.name) + "'";
        region_a_prime.area = std::abs(area_a_prime);
        region_a_prime.black_line_value = construction.black_line_value;
        region_a_prime.start_R = construction.critical_R;
        region_a_prime.end_R = construction.intersection2_R;
        regions.push_back(region_a_prime);
        
        std::cout << "  区域 " << region_a_prime.name << ": R=[" << construction.critical_R 
                  << ", " << construction.intersection2_R << "], 面积=" << region_a_prime.area 
                  << " (积分 f_black - f_red)" << std::endl;
        
        return regions;
    }
    
    // 积分计算面积
    double integrateArea(double start_R, double end_R, double black_line_value, bool invert) {
        double total_area = 0.0;
        
        for (size_t i = 0; i < red_data.size() - 1; ++i) {
            const DataPoint& p1 = red_data[i];
            const DataPoint& p2 = red_data[i + 1];
            
            if (p2.R <= start_R) continue;
            if (p1.R >= end_R) break;
            
            double segment_start_R = std::max(p1.R, start_R);
            double segment_end_R = std::min(p2.R, end_R);
            
            if (segment_start_R < segment_end_R) {
                // 线性插值
                double t1 = (segment_start_R - p1.R) / (p2.R - p1.R);
                double t2 = (segment_end_R - p1.R) / (p2.R - p1.R);
                
                double red_height1 = p1.f_red + t1 * (p2.f_red - p1.f_red);
                double red_height2 = p1.f_red + t2 * (p2.f_red - p1.f_red);
                
                double height1, height2;
                if (invert) {
                    height1 = black_line_value - red_height1;
                    height2 = black_line_value - red_height2;
                } else {
                    height1 = red_height1 - black_line_value;
                    height2 = red_height2 - black_line_value;
                }
                
                double segment_area = 0.5 * (height1 + height2) * (segment_end_R - segment_start_R);
                total_area += segment_area;
            }
        }
        
        return total_area;
    }

public:
    // 计算所有区域面积并比较对称区域
    void calculateAndCompareAllAreas() {
        std::vector<MaxwellConstruction> constructions = detectMaxwellConstructions();
        
        std::cout << "\n=== 麦克斯韦构造区域面积计算 ===" << std::endl;
        
        std::vector<Region> all_regions;
        
        // 收集所有区域
        for (const auto& construction : constructions) {
            all_regions.insert(all_regions.end(), 
                             construction.regions.begin(), 
                             construction.regions.end());
        }
        
        // 输出所有区域面积
        std::cout << "\n所有区域面积汇总:" << std::endl;
        for (const auto& region : all_regions) {
            std::cout << "区域 " << region.name << ": 面积 = " << region.area 
                      << " (黑线值 = " << region.black_line_value 
                      << ", R范围 = [" << region.start_R << ", " << region.end_R << "])" << std::endl;
        }
        
        // 比较对称区域
        compareSymmetricAreas(constructions);
        
        // 保存结果到文件
        saveResultsToFile(all_regions, "maxwell_areas_final.csv");
    }
    
    // 比较对称区域面积
    void compareSymmetricAreas(const std::vector<MaxwellConstruction>& constructions) {
        std::cout << "\n=== 对称区域面积比较 ===" << std::endl;
        
        for (const auto& construction : constructions) {
            if (construction.regions.size() == 2) {
                const Region& region_a = construction.regions[0];
                const Region& region_a_prime = construction.regions[1];
                
                double difference = std::abs(region_a.area - region_a_prime.area);
                double relative_error = difference / ((region_a.area + region_a_prime.area) / 2.0) * 100.0;
                
                std::cout << "构造 " << construction.name << ":" << std::endl;
                std::cout << "  " << region_a.name << " 面积: " << region_a.area 
                          << " (R: " << region_a.start_R << "→" << region_a.end_R << ")" << std::endl;
                std::cout << "  " << region_a_prime.name << " 面积: " << region_a_prime.area 
                          << " (R: " << region_a_prime.start_R << "→" << region_a_prime.end_R << ")" << std::endl;
                std::cout << "  面积差: " << difference << std::endl;
                std::cout << "  相对误差: " << relative_error << "%" << std::endl;
                
                if (relative_error < 1.0) {
                    std::cout << "  ✓ 面积基本相等 (符合麦克斯韦构造)" << std::endl;
                } else {
                    std::cout << "  ⚠ 面积有显著差异" << std::endl;
                }
                std::cout << std::endl;
            }
        }
    }
    
    // 保存结果到文件
    void saveResultsToFile(const std::vector<Region>& regions, const std::string& filename) {
        std::ofstream outfile(filename);
        
        if (!outfile.is_open()) {
            std::cerr << "错误：无法创建输出文件 " << filename << std::endl;
            return;
        }
        
        // 写入CSV标题
        outfile << "Region,Area,Black_Line_Value,Start_R,End_R,Integration_Type" << std::endl;
        
        // 写入数据
        for (const auto& region : regions) {
            std::string integration_type = (region.name.back() == '\'') ? 
                                          "f_black - f_red" : "f_red - f_black";
            outfile << region.name << "," << region.area << "," 
                    << region.black_line_value << "," << region.start_R << "," 
                    << region.end_R << "," << integration_type << std::endl;
        }
        
        outfile.close();
        std::cout << "详细结果已保存到: " << filename << std::endl;
    }
};

int main() {
    // CSV文件名
    std::string red_file = "/home/tyt/project/Single-chain/opt+R/WLC+R_data.csv";    // 红线数据文件
    std::string blue_file = "/home/tyt/project/Single-chain/opt-force/WLC+force_results.csv";  // 蓝线数据文件
    
    // 从CSV文件读取数据
    std::vector<DataPoint> red_data;
    std::vector<std::pair<double, double>> blue_data;
    
    if (!MaxwellConstructionIntegrator::readDataFromCSV(red_file, blue_file, red_data, blue_data)) {
        std::cerr << "错误：无法读取数据或数据为空" << std::endl;
        return 1;
    }
    
    // 创建积分器
    MaxwellConstructionIntegrator integrator(red_data, blue_data);
    
    // 计算所有区域面积并比较
    integrator.calculateAndCompareAllAreas();
    
    return 0;
}
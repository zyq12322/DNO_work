#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "DNO_calc_OE.h"

int main() {
    // total number of points
    size_t N = 64;
    // length of computational domain
    const double L = 2.0 * M_PI;          // 物理域长度
    // delta x
    double dx = L / N;
    // amplitude of airy wave
    double a = 0.1;
    // gravity
    double g = 9.81;
    // wave number
    double k = 1;
    // actrul water depth
    double h = 11;
    // mean_depth
    double mean_depth = 10;
    // circle frequency
    double omega = std::sqrt(g * k * std::tanh(k * h));

    // construct global FFT / Dealiaser object
    fft_obj   = std::make_unique<fft_op>(N);
    dealiaser = std::make_unique<Dealiaser>(N, 0);   // 1D

    // x grid
    std::vector<double> x_grid = linspace(0, L, N, false);
    // k grid
    std::vector<double> freq_grid = fftfreq(N, dx);        // 空间频率，单位 1/m
    std::vector<double> k_grid(N);
    for (size_t i = 0; i < N; ++i) {
        k_grid[i] = 2.0 * M_PI * freq_grid[i]; // 角波数，单位 rad/m
    }
    // eta
    std::vector<double> eta_real(N);
    for (size_t i = 0; i < N; ++i) {
        eta_real[i] = a * std::cos(k * x_grid[i]);
    }
    std::vector<std::complex<double>> eta = double_to_complex_real(eta_real);
    std::vector<std::complex<double>> eta_hat = fft_obj->fft(eta);
    // xi- velocity potential on surface
    std::vector<double> xi_real(N);
    for (size_t i = 0; i < N; ++i) {
        xi_real[i] = a * g / omega * std::cosh(k * (eta_real[i] + h)) / std::cosh(k * h) * std::sin(k * x_grid[i]);
    }
    std::vector<std::complex<double>> xi = double_to_complex_real(xi_real);
    std::vector<std::complex<double>> xi_hat = fft_obj->fft(xi);
    // G^E - exact solution for DNO
    std::vector<double> GE_real(N);
    for (size_t i = 0; i < N; ++i) {
        GE_real[i] = a * g / omega / std::cosh(k * h) * ( k * std::sinh(k * (eta_real[i] + h)) * std::sin(k * x_grid[i])
                    + a * k * std::sin(k * x_grid[i]) * k * std::cosh(k * (eta_real[i] + h)) * std::cos(k * x_grid[i]) );
    }
    std::vector<std::complex<double>> GE = double_to_complex_real(GE_real);
    std::vector<std::complex<double>> GE_hat = fft_obj->fft(GE);

    // beta (flat bottom in this example)
    std::vector<double> beta_real(N, mean_depth - h);
    std::vector<std::complex<double>> beta = double_to_complex_real(beta_real);
    std::vector<std::complex<double>> beta_hat = fft_obj->fft(beta);
    // gamma
    std::vector<std::complex<double>> gamma_hat(N, std::complex<double>(0.0, 0.0));

    bool dealiase_flag = false;

    // ===================== 新增：阶数扫描循环 =====================
    const int M_MAX = 15;                               // DNO_M 最大阶数（横轴范围 0..8）
    const std::vector<int> Mb_list = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};  // 不同的 DNO_Mb（每条图线）

    std::ofstream fout("dno_error_results.txt");
    fout << "# DNO_M  DNO_Mb  RMSE  Relative_L2\n";

    for (int mb : Mb_list) {               // 外层：不同的 DNO_Mb
        for (int m = 0; m <= M_MAX; ++m) { // 内层：DNO_M 从 0 扫到 M_MAX
            int DNO_M  = m;
            int DNO_Mb = mb;

            // 每个 (DNO_M, DNO_Mb) 组合都重新初始化工作区
            // （阶乘表、D_pow_hat、eta/beta 幂次表都会按当前阶数重建）
            init_DNO_workspace_2d(DNO_M, DNO_Mb, N, k_grid, eta_hat, beta_hat, dealiase_flag);

            // DNO result
            std::vector<std::complex<double>> GC_hat(N, std::complex<double>(0.0, 0.0));
            partial_eta_partial_t_2d(xi_hat, gamma_hat, GC_hat, mean_depth, DNO_M, DNO_Mb, dealiase_flag);

            auto [rmse, rel_err] = compute_l2_error(GC_hat, GE_hat);

            // 写入结果文件：DNO_M  DNO_Mb  RMSE  Relative_L2
            fout << DNO_M << "  " << DNO_Mb << "  "
                 << std::scientific << std::setprecision(8)
                 << rmse << "  " << rel_err << '\n';

            std::cout << "DNO_M = " << DNO_M << "  DNO_Mb = " << DNO_Mb
                      << "  RMSE = " << rmse
                      << "  Relative_L2 = " << rel_err << '\n';
        }
    }
    fout.close();
    std::cout << "All results written to dno_error_results.txt\n";
    return 0;
}
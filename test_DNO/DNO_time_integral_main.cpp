#include "DNO_time_integral.h"
#include <iostream>

int main() {
    const size_t N = 4096;
    const double L = 64.0 * M_PI;
    const double dx = L / N;
    const double a = 0.1;
    const double g = 9.81;
    const double k = 1.0;
    const double h = 11.0; // actual depth
    const double mean_depth = 10.0;
    // linear dispersion relation
    const double omega = std::sqrt(g * k * std::tanh(k * h));
    const double period = 2 * M_PI / omega;
    const double H_gT2 = 2 * a / g / period / period;
    const double d_gT2 = h / g / period / period;
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "period = " << period << " s\n";
    std::cout << "H/(gT^2) = " << H_gT2 << "\n";
    std::cout << "d/(gT^2) = " << d_gT2 << "\n";

    // 1. 全局 FFT / Dealiaser（本模块依赖它们）
    fft_obj   = std::make_unique<fft_op>(N);
    dealiaser = std::make_unique<Dealiaser>(N, 0);

    // 2. 角波数轴
    std::vector<double> x_grid = linspace(0, L, N, false);
    std::vector<double> freq = fftfreq(N, dx);
    std::vector<double> k_axis(N);
    for (size_t i = 0; i < N; ++i) k_axis[i] = 2.0 * M_PI * freq[i];

    // 3. 预分配 DNO 工作区（必须！）
    const int DNO_M  = 5;
    const int DNO_Mb = 5;
    resize_DNO_workspace(DNO_M, DNO_Mb, N);

    // 4. 初始 η、ξ
    std::vector<double> eta_phy(N), xi_phy(N);

    // // initial condition - Airy wave
    // for (size_t i = 0; i < N; ++i) {
    //     eta_phy[i] = a * std::cos(k * x_grid[i]);
    //     xi_phy[i]  = a * g / omega * std::cosh(k * (eta_phy[i] + h))
    //                / std::cosh(k * h) * std::sin(k * x_grid[i]);
    // }

    // ---- 初始条件：Stokes 二阶波（严格依据提供的图片公式），t = 0 ----
    // 波面 theta = kx - wt，在 t = 0 时 theta = kx
    const double sigma = std::tanh(k * h);
    const double sh = std::sinh(k * h);
    
    // 1. 计算 eta 的三个部分
    // 常数项（平均水位下降 / set-down）
    const double eta0 = - (k * a * a) / (2.0 * std::sinh(2.0 * k * h)); 
    
    // 二阶谐波系数
    const double eta2_coef = (k * a * a / 4.0) * (3.0 - sigma * sigma) / (sigma * sigma * sigma);

    // 2. 计算势 xi 的系数
    // 一阶势系数
    const double xi1_coef = a * g / omega;
    // 二阶势系数：注意分母是 8 * sinh^4(kd)
    const double xi2_coef = (3.0 * a * a * omega) / (8.0 * std::pow(sh, 4)); 

    for (size_t i = 0; i < N; ++i) {
        const double theta = k * x_grid[i];      // t = 0

        // η：常数项 + 一阶 + 二阶
        eta_phy[i] = eta0 
                   + a * std::cos(theta)
                   + eta2_coef * std::cos(2.0 * theta);

        // ξ (即表面势 Φ|_{z=η})：一阶 + 二阶
        // 注意图中的 z+d，这里的 z 是自由表面高程 eta，d 是水深 h
        const double z_plus_d = eta_phy[i] + h;

        // 一阶 Airy 势
        const double xi1 = xi1_coef 
                         * std::cosh(k * z_plus_d) / std::cosh(k * h) 
                         * std::sin(theta);
                         
        // 二阶 Stokes 势
        const double xi2 = xi2_coef 
                         * std::cosh(2.0 * k * z_plus_d) 
                         * std::sin(2.0 * theta);

        xi_phy[i] = xi1 + xi2;
    }

    DNOState state;
    state.eta_hat = fft_obj->fft(double_to_complex_real(eta_phy));
    state.xi_hat  = fft_obj->fft(double_to_complex_real(xi_phy));

    DNOSystemParams p;
    p.h0            = mean_depth;
    p.g             = g;
    p.DNO_M         = DNO_M;
    p.DNO_Mb        = DNO_Mb;
    p.dealiase_flag = true;
    p.k_axis        = k_axis;

    // 5. β、γ 时间回调（示例：β 有微小周期起伏）
    // 5. β、γ 时间回调：底部为随时间平滑上升的高斯突起，最终稳定
    //
    //    β(x,t) = (mean_depth - h) + A(t) * exp(-((x-x0)/σ)^2)
    //    γ(x,t) = dA/dt   * exp(-((x-x0)/σ)^2)
    //
    //    A(t)   = A∞/2 * (1 + tanh((t - t_r)/τ))
    //    A'(t)  = A∞/(2τ) * sech^2((t - t_r)/τ)
    //
    // const double A_inf   = 1.0;         // 突起最终高度
    // const double sigma_b = 0.8;         // 高斯宽度 (rad)
    // const double x0_b    = L / 2.0;        // 突起中心
    // const double t_r     = 0.2;         // 上升半程时刻
    // const double tau_b   = 0.05;         // 上升时间尺度

    // p.bc_func = [&, A_inf, sigma_b, x0_b, t_r, tau_b](
    //                 double t,
    //                 std::vector<std::complex<double>>& beta_hat,
    //                 std::vector<std::complex<double>>& gamma_hat) {
    //     // A(t) 与 A'(t)
    //     const double s     = (t - t_r) / tau_b;
    //     const double th    = std::tanh(s);
    //     const double sech2 = 1.0 - th * th;             // sech²(s)

    //     const double A     = 0.5 * A_inf * (1.0 + th);
    //     const double dA_dt = 0.5 * A_inf * sech2 / tau_b;

    //     std::vector<double> b_phy(N), g_phy(N);
    //     for (size_t i = 0; i < N; ++i) {
    //         const double u     = (x_grid[i] - x0_b) / sigma_b;
    //         const double gauss = std::exp(-u * u);

    //         b_phy[i] = -1 + A     * gauss;
    //         g_phy[i] = dA_dt * gauss;
    //     }

    //     beta_hat  = fft_obj->fft(double_to_complex_real(b_phy));
    //     gamma_hat = fft_obj->fft(double_to_complex_real(g_phy));
    // };


    // 平坦底部
    p.bc_func = [&](double t,
                    std::vector<std::complex<double>>& beta_hat,
                    std::vector<std::complex<double>>& gamma_hat) {
        std::vector<double> beta_phy(N);
        for (size_t i = 0; i < N; ++i) {
            beta_phy[i] = (mean_depth - h);
        }
        beta_hat  = fft_obj->fft(double_to_complex_real(beta_phy));
        std::fill(gamma_hat.begin(), gamma_hat.end(),
                std::complex<double>(0.0, 0.0));
    };
    
    // p.bc_func = [&](double t,
    //                 std::vector<std::complex<double>>& beta_hat,
    //                 std::vector<std::complex<double>>& gamma_hat) {
    //     std::vector<double> b_phy(N), g_phy(N);
    //     for (size_t i = 0; i < N; ++i) {
    //         b_phy[i] = (mean_depth - h)
    //                  + 0.01 * std::sin(0.3 * t) * std::cos(k * x_grid[i]);
    //         g_phy[i] = 0.01 * 0.3 * std::cos(0.3 * t) * std::cos(k * x_grid[i]);
    //     }
    //     beta_hat  = fft_obj->fft(double_to_complex_real(b_phy));
    //     gamma_hat = fft_obj->fft(double_to_complex_real(g_phy));
    // };

    // 5.5 输出记录：每 0.01 时间单位写出一次 η、ξ、β，Ek/Ep/I/V
    DNOOutputConfig out_cfg;
    out_cfg.enabled  = true;
    out_cfg.interval = 0.001;
    out_cfg.x_axis   = x_grid;          // 长度必须 = N
    out_cfg.eta_file  = "eta.txt";
    out_cfg.xi_file   = "xi.txt";
    out_cfg.beta_file = "beta.txt";
    out_cfg.EVI_file = "EVI.txt";

    DNOOutputRecorder recorder(out_cfg, /*t0=*/0.0);

    // 6. 时间推进
    const size_t n = dno_time_integrate(
        state, /*t0=*/0.0, /*t_end=*/5, /*dt_init=*/1e-4,
        /*rtol=*/5e-12, /*atol=*/1e-12, p,
        /*max_steps=*/1000000,
        &recorder);

    std::cout << "Accepted steps: " << n << "\n";
    return 0;
}
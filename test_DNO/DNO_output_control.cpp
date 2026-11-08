// DNO_output_control.cpp
#include "DNO_output_control.h"
#include "DNO_calc_OE.h"        // partial_eta_partial_t_2d, D_pow_hat, init_DNO_workspace_2d
#include "DNO_time_integral.h"  // DNOSystemParams 的完整定义
#include "var_op_function.h"    // fft_obj

#include <stdexcept>
#include <iomanip>

// global object - external initialization required
// 全局控制数据结构体
DNOSystemParams g_dno_params;

// global object - external initialization required
// 此项必须在全局设置g_dno_params全部正确填充了之后初始化
std::unique_ptr<DNOOutputRecorder> recorder;


DNOOutputRecorder::DNOOutputRecorder()
{
    // 允许输出的条件是否都具备
    record_ = g_dno_params.enable_output
           && g_dno_params.output_interval > 0.0
           && !g_dno_params.x_axis.empty();

    if (!record_) {
        return;
    }

    if (!fft_obj || !dealiaser) {
        throw std::runtime_error(
            "DNOOutputRecorder: fft_obj or dealiaser is not initialized");
    }

    eta_out_.open(g_dno_params.eta_file);
    xi_out_.open(g_dno_params.xi_file);
    beta_out_.open(g_dno_params.beta_file);
    EVI_out_.open(g_dno_params.EVI_file);
    if (!eta_out_ || !xi_out_ || !beta_out_ || !EVI_out_) {
        throw std::runtime_error(
            "DNOOutputRecorder: cannot open output files");
    }

    N_ = g_dno_params.x_axis.size();

    eta_phy_.resize(N_);
    xi_phy_.resize(N_);
    beta_phy_.resize(N_);
    beta_hat_.resize(N_);
    gamma_hat_.resize(N_);

    G_hat_.resize(N_);
    G_phy_.resize(N_);

    eta_x_hat_.resize(N_);
    eta_x_phy_.resize(N_);

    xi_G_hat_.resize(N_);
    eta2_hat_.resize(N_);
    xi_etax_hat_.resize(N_);
    xi_G_phy_.resize(N_);
    eta2_phy_.resize(N_);
    xi_etax_phy_.resize(N_);

    f_Ek_.resize(N_);
    f_Ep_.resize(N_);
    f_I_.resize(N_);
    f_V_.resize(N_);

    eta_out_  << std::setprecision(16);
    xi_out_   << std::setprecision(16);
    beta_out_ << std::setprecision(16);
    EVI_out_ << std::setprecision(16);

    // eta/xi/beta表头: 0, x0, x1, ..., x_{N-1}
    eta_out_  << 0;
    xi_out_   << 0;
    beta_out_ << 0;
    for (double x : g_dno_params.x_axis) {
        eta_out_  << "," << x;
        xi_out_   << "," << x;
        beta_out_ << "," << x;
    }
    eta_out_  << "\n";
    xi_out_   << "\n";
    beta_out_ << "\n";

    next_record_t_ = g_dno_params.t0;
}

void DNOOutputRecorder::record(
    double t,
    const std::vector<std::complex<double>>& eta_hat,
    const std::vector<std::complex<double>>& xi_hat)
{
    if (!record_) return;
    // check size
    const size_t N = g_dno_params.x_axis.size();
    
    if (eta_hat.size() != N || xi_hat.size() != N || g_dno_params.k_axis.size() != N) {
        throw std::invalid_argument(
            "DNOOutputRecorder::record: input size mismatch with x_axis");
    }
    if (!fft_obj || !dealiaser) {
        throw std::runtime_error(
            "DNOOutputRecorder::record: fft_obj or dealiaser is not initialized");
    }

    // 防御：若外部 N 变了（正常不会），重新分配
    if (N_ != N) {
        N_ = N;
        eta_phy_.resize(N);   xi_phy_.resize(N);   beta_phy_.resize(N);
        beta_hat_.resize(N);    gamma_hat_.resize(N);
        G_hat_.resize(N);     G_phy_.resize(N);
        eta_x_hat_.resize(N); eta_x_phy_.resize(N);
        xi_G_hat_.resize(N);  eta2_hat_.resize(N); xi_etax_hat_.resize(N);
        xi_G_phy_.resize(N);  eta2_phy_.resize(N); xi_etax_phy_.resize(N);
        f_Ek_.resize(N); f_Ep_.resize(N); f_I_.resize(N); f_V_.resize(N);
    }
    // 填充零防止bc_func未覆盖旧值
    std::fill(beta_hat_.begin(), beta_hat_.end(), std::complex<double>(0.0, 0.0));
    std::fill(gamma_hat_.begin(), gamma_hat_.end(), std::complex<double>(0.0, 0.0));
    g_dno_params.bc_func(t, beta_hat_, gamma_hat_);

    // eta/xi/beta 频谱 -> 物理空间
    eta_phy_  = fft_obj->ifft(eta_hat);
    xi_phy_   = fft_obj->ifft(xi_hat);
    beta_phy_ = fft_obj->ifft(beta_hat_);


    init_DNO_workspace_2d(g_dno_params.DNO_M, g_dno_params.DNO_Mb, N, g_dno_params.k_axis,
                          eta_hat, beta_hat_, g_dno_params.dealiase_flag);

    partial_eta_partial_t_2d(xi_hat, gamma_hat_, G_hat_,
                             g_dno_params.h0, g_dno_params.DNO_M, g_dno_params.DNO_Mb, g_dno_params.dealiase_flag);

    // --- 2) 逆变换 ---
    G_phy_ = fft_obj->ifft(G_hat_);

    // --- 3) η_x 的Fourier Sapce Value ---
    for (size_t i = 0; i < N; ++i) {
        eta_x_hat_[i] = ImUnit * D_pow_hat[1][i] * eta_hat[i];
    }
    // \partial_x \eta Fourier Space -> Physical space
    eta_x_phy_ = fft_obj->ifft(eta_x_hat_);

    // --- 4) 数值积分（周期网格，使用 integrate 工具 + PeriodicTrap） ---
    //     x_axis 为 FFT 周期网格：N 点覆盖 [0,L)，dx = L/N。
    //     被积函数（η, ξ, G 等）均为光滑周期函数，PeriodicTrap 对其具谱精度。
    //     非线性乘积 (ξ·G, η·η, ξ·η_x) 按 dealiase_flag 分情况处理。
    const double dx = g_dno_params.x_axis[1] - g_dno_params.x_axis[0];
    const QuadMethod qm = QuadMethod::PeriodicTrap;

    double Ek = 0.0, Ep = 0.0, I = 0.0, V = 0.0;

    if (!g_dno_params.dealiase_flag) {
        // ---- 物理空间逐点相乘，再周期积分 ----
        for (size_t i = 0; i < N; ++i) {
            const double eta_v  = eta_phy_[i].real();
            const double beta_v = beta_phy_[i].real();
            const double xi_v   = xi_phy_[i].real();
            const double G_v    = G_phy_[i].real();
            const double etax_v = eta_x_phy_[i].real();

            f_Ek_[i] = 0.5 * xi_v * G_v;
            f_Ep_[i] = 0.5 * g_dno_params.g * eta_v * eta_v;
            f_I_[i]  =       xi_v * etax_v;
            f_V_[i]  = (eta_v - beta_v + g_dno_params.h0);
        }
    } else {
        // ---- 谱空间通过 dealiaser 计算乘积，再逆变换到物理空间积分 ----

        (*dealiaser)(xi_hat.data(), G_hat_.data(),      false, false, xi_G_hat_.data());
        (*dealiaser)(eta_hat.data(), eta_hat.data(),   false, false, eta2_hat_.data());
        (*dealiaser)(xi_hat.data(), eta_x_hat_.data(),  false, false, xi_etax_hat_.data());

        xi_G_phy_    = fft_obj->ifft(xi_G_hat_);
        eta2_phy_    = fft_obj->ifft(eta2_hat_);
        xi_etax_phy_ = fft_obj->ifft(xi_etax_hat_);

        for (size_t i = 0; i < N; ++i) {
            f_Ek_[i] = 0.5 *       xi_G_phy_[i].real();
            f_Ep_[i] = 0.5 * g_dno_params.g * eta2_phy_[i].real();
            f_I_[i]  =             xi_etax_phy_[i].real();
            f_V_[i]  = eta_phy_[i].real() - beta_phy_[i].real() + g_dno_params.h0;
        }
        
    }
    // 数值积分
    Ek = integrate(f_Ek_, dx, qm);
    Ep = integrate(f_Ep_, dx, qm);
    I  = integrate(f_I_,  dx, qm);
    V  = integrate(f_V_,  dx, qm);

    eta_out_  << t;
    xi_out_   << t;
    beta_out_ << t;
    EVI_out_ << t;

    for (size_t i = 0; i < N; ++i) {
        eta_out_  << "," << eta_phy_[i].real();
        xi_out_   << "," << xi_phy_[i].real();
        beta_out_ << "," << beta_phy_[i].real();
    }

    EVI_out_  << "," << Ek;
    EVI_out_  << "," << Ep;
    EVI_out_  << "," << I;
    EVI_out_  << "," << V;
    

    eta_out_  << "\n";
    xi_out_   << "\n";
    beta_out_ << "\n";
    EVI_out_ << "\n";

    eta_out_.flush();
    xi_out_.flush();
    beta_out_.flush();
    EVI_out_.flush();

    next_record_t_ = t + g_dno_params.output_interval;
}
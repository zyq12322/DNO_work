// DNO_output_control.h
#ifndef DNO_OUTPUT_CONTROL_H
#define DNO_OUTPUT_CONTROL_H

#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include <cstddef>
#include <functional>

// ---------------------------------------------------------------------
// β、γ 时间回调：给定时刻 t，填充 beta_hat 和 gamma_hat({\partial_t \eta} _hat)
//
// 说明：
//   - 若 β、γ 为常数，回调里直接填入预置值即可；
//   - 若 β 随时间变化且 γ = ∂β/∂t 有解析表达式，可在同一回调中计算；
//   - 若 γ 由外部数值方法提供，也在这里统一返回。
// ---------------------------------------------------------------------
using DNOBottomFunc = std::function<void(
    double t,
    std::vector<std::complex<double>>& beta_hat,
    std::vector<std::complex<double>>& gamma_hat)>;

// ---------------------------------------------------------------------
// 系统参数：物理常数、阶数、波数轴、β/γ 回调
//
// 调用者必须提前：
//   1. 创建全局 fft_obj / dealiaser；
//   2. 调用 resize_DNO_workspace(DNO_M, DNO_Mb, N) 一次。
// 本模块不会主动 resize。
// ---------------------------------------------------------------------
struct DNOSystemParams {
    double h0;                      // 平均水深
    double g;                       // 重力加速度
    int    DNO_M;                   // η 最大幂次
    int    DNO_Mb;                  // β 最大幂次
    bool   dealiase_flag;           // 是否使用反混叠
    std::vector<double> k_axis;     // 角波数轴 (rad/m)，长度 N
    DNOBottomFunc bc_func;              // β、γ 时间回调
};

// ---------------------------------------------------------------------
// DNO 输出配置
//
// 每隔 interval 时间单位，把 η、ξ、β EVI 的物理空间值写入文件。
// eta/xi/beta文件格式：
//   第 1 行 : 0, x0, x1, ..., x_{N-1} (0, x_axis[0], ...)
//   第 k 行 : t_i, f(x0), f(x1), ..., f(x_{N-1}) , i = 0, 1 , ...
// EVI文件格式(total in computational domain):
//   t_i, E_k, E_p, I, V
// ---------------------------------------------------------------------
struct DNOOutputConfig {
    bool                enabled   = false;// flag for recording or not
    double              interval  = 0.0;         // 记录时间间隔
    std::vector<double> x_axis;                  // 物理网格（长度 N）
    std::string         eta_file  = "eta.txt";
    std::string         xi_file   = "xi.txt";
    std::string         beta_file = "beta.txt";
    std::string         EVI_file = "EVI.txt"; // energy/volume/momentum
};

// ---------------------------------------------------------------------
// DNO 输出记录器
//
// 使用方式：
//     DNOOutputRecorder rec(cfg, /*t0=*/0.0);          // 构造即建文件 + 写表头
//     rec.record(t0, eta_hat, xi_hat, beta_hat);       // 记录初始时刻（可选）
//     while (...) {
//         ... 积分一步 ...
//         if (rec.due(t)) {
//             std::vector<std::complex<double>> beta_hat(N), gamma_hat(N);
//             p.bc_func(t, beta_hat, gamma_hat);
//             rec.record(t, state.eta_hat, state.xi_hat, beta_hat, gamma_hat, p);
//         }
//     }
//
// 说明：
//   - 若 cfg.enabled==false / interval<=0 / x_axis 为空，则记录器处于
//     disabled 状态，due() 恒返回 false，record() 不做任何事。
//   - 要求全局 fft_obj 已初始化（用于 ifft 到物理空间）。
//   - 三个输入向量必须与 x_axis 长度一致。
// ---------------------------------------------------------------------
class DNOOutputRecorder {
public:
    DNOOutputRecorder() = default;

    // 构造时若启用，则打开三个文件并写入表头行。
    // t0 用作首次记录时刻（next_record_t_ 初始化为 t0）。
    DNOOutputRecorder(const DNOOutputConfig& cfg, double t0);

    bool   enabled()   const noexcept { return record_; }

    // 是否应该在时刻 t 调用 record()（带 1e-12 容差）
    bool due(double t) const noexcept {
        return record_ && t >= next_record_t_ - 1e-12;
    }

    // 写出时刻 t 的一条记录，之后把 next_record_t_ 推进到 t + interval。
    // eta_hat / xi_hat / beta_hat / gamma_hat 均为傅里叶空间系数，长度 = N。
    void record(double t,
                const std::vector<std::complex<double>>& eta_hat,
                const std::vector<std::complex<double>>& xi_hat,
                const std::vector<std::complex<double>>& beta_hat,
                const std::vector<std::complex<double>>& gamma_hat,
                const DNOSystemParams& p);

private:
    DNOOutputConfig cfg_;
    bool            record_         = false;
    double          next_record_t_  = 0.0;

    std::ofstream   eta_out_;
    std::ofstream   xi_out_;
    std::ofstream   beta_out_;
    std::ofstream   EVI_out_;

    // ---- 复用的工作缓冲区（构造时按 N 分配）----
    size_t N_ = 0;

    std::vector<std::complex<double>> eta_phy_;
    std::vector<std::complex<double>> xi_phy_;
    std::vector<std::complex<double>> beta_phy_;

    std::vector<std::complex<double>> G_hat_;
    std::vector<std::complex<double>> G_phy_;

    std::vector<std::complex<double>> eta_x_hat_;
    std::vector<std::complex<double>> eta_x_phy_;

    // dealias 分支专用
    std::vector<std::complex<double>> xi_G_hat_;
    std::vector<std::complex<double>> eta2_hat_;
    std::vector<std::complex<double>> xi_etax_hat_;
    std::vector<std::complex<double>> xi_G_phy_;
    std::vector<std::complex<double>> eta2_phy_;
    std::vector<std::complex<double>> xi_etax_phy_;

    // 积分被积函数（double 序列）
    std::vector<double> f_Ek_, f_Ep_, f_I_, f_V_;
};

#endif // DNO_OUTPUT_CONTROL_H
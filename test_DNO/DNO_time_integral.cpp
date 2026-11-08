#include "DNO_time_integral.h"


#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <iostream>
#include <iomanip>

namespace {

    // 持久缓存：本线程内唯一，避免每阶段分配
    thread_local RK45BufferCache rk45_cache;

    // ---------------------------------------------------------------------
    // 求 DNO 系统在时刻 t 的右端项
    //
    //   dη/dt = G(β(t), η)[ξ, γ(t)]
    //   dξ/dt = -g η - 0.5 (ξ_x² - G² - 2 η_x ξ_x G) / (1 + η_x²)
    //
    // 关键：每次调用都先用当前 η、该时刻 β 刷新全局工作区，
    // 因为 partial_xi_partial_t_2d 依赖全局 eta_pow_hat[1]。
    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------
    // 求 DNO 系统在时刻 t 的右端项
    // ---------------------------------------------------------------------
    void compute_dno_rhs(
        double t,
        const std::vector<std::complex<double>>& eta_hat,// eta at time t
        const std::vector<std::complex<double>>& xi_hat,// xi at time t
        const DNOSystemParams& p,
        std::vector<std::complex<double>>& deta_dt,
        std::vector<std::complex<double>>& dxi_dt,
        DNOBufferCache& cache)   // ← 新增
    {
        const size_t N = eta_hat.size();
        if (N == 0)
            throw std::invalid_argument("compute_dno_rhs: empty state");
        if (xi_hat.size() != N)
            throw std::invalid_argument("compute_dno_rhs: xi size mismatch");
        if (p.k_axis.size() != N)
            throw std::invalid_argument("compute_dno_rhs: k_axis size mismatch");
        if (!p.bc_func)
            throw std::invalid_argument("compute_dno_rhs: bc_func not set");

        // 复用缓存里的 beta_hat / gamma_hat
        std::fill(cache.beta_hat.begin(),  cache.beta_hat.end(),  0.0);
        std::fill(cache.gamma_hat.begin(), cache.gamma_hat.end(), 0.0);
        // 1. 取该时刻的 β(t)、γ(t)
        p.bc_func(t, cache.beta_hat, cache.gamma_hat);

        // 2. 用当前 η、该时刻 β 刷新全局工作区
        //    （更新eta_pow_hat / beta_pow_hat / D_pow_hat，factorial_list不变）
        init_DNO_workspace_2d(p.DNO_M, p.DNO_Mb, N, p.k_axis,
                            eta_hat, cache.beta_hat, p.dealiase_flag);

        partial_eta_partial_t_2d(xi_hat, cache.gamma_hat, deta_dt,
                                p.h0, p.DNO_M, p.DNO_Mb, p.dealiase_flag,
                                &cache);

        partial_xi_partial_t_2d(xi_hat, deta_dt, dxi_dt, p.g,
                                p.DNO_M, p.DNO_Mb, p.dealiase_flag,
                                &cache);
    }

} // namespace

// =====================================================================
// Dormand-Prince RK45(5(4)) 单步
// =====================================================================
RK45AttemptResult rk45_step(
    double t, double dt,
    const DNOSystemParams& p,
    DNOState& state,
    double rtol, double atol,
    double safety, double min_factor, double max_factor)
{
    const size_t N = state.eta_hat.size();

    
    rk45_cache.ensure_size(N, p.DNO_M, p.DNO_Mb);

    auto& k1e = rk45_cache.k1e;  auto& k1x = rk45_cache.k1x;
    auto& k2e = rk45_cache.k2e;  auto& k2x = rk45_cache.k2x;
    auto& k3e = rk45_cache.k3e;  auto& k3x = rk45_cache.k3x;
    auto& k4e = rk45_cache.k4e;  auto& k4x = rk45_cache.k4x;
    auto& k5e = rk45_cache.k5e;  auto& k5x = rk45_cache.k5x;
    auto& k6e = rk45_cache.k6e;  auto& k6x = rk45_cache.k6x;
    auto& k7e = rk45_cache.k7e;  auto& k7x = rk45_cache.k7x;
    auto& eta_tmp = rk45_cache.eta_tmp;  auto& xi_tmp = rk45_cache.xi_tmp;
    auto& eta_err = rk45_cache.eta_err;  auto& xi_err = rk45_cache.xi_err;

    // ---- 系数（保持不变）----
    constexpr double c2 = 1.0/5.0, c3 = 3.0/10.0, c4 = 4.0/5.0;
    constexpr double c5 = 8.0/9.0, c6 = 1.0,      c7 = 1.0;
    constexpr double a21 = 1.0/5.0;
    constexpr double a31 = 3.0/40.0,  a32 = 9.0/40.0;
    constexpr double a41 = 44.0/45.0, a42 = -56.0/15.0, a43 = 32.0/9.0;
    constexpr double a51 = 19372.0/6561.0, a52 = -25360.0/2187.0,
                     a53 = 64448.0/6561.0,  a54 = -212.0/729.0;
    constexpr double a61 = 9017.0/3168.0,  a62 = -355.0/33.0,
                     a63 = 46732.0/5247.0, a64 = 49.0/176.0,
                     a65 = -5103.0/18656.0;
    constexpr double a71 = 35.0/384.0, a73 = 500.0/1113.0,
                     a74 = 125.0/192.0, a75 = -2187.0/6784.0,
                     a76 = 11.0/84.0;
    constexpr double b1 = a71, b3 = a73, b4 = a74, b5 = a75, b6 = a76;
    constexpr double b1s = 5179.0/57600.0,
                     b3s = 7571.0/16695.0,
                     b4s = 393.0/640.0,
                     b5s = -92097.0/339200.0,
                     b6s = 187.0/2100.0,
                     b7s = 1.0/40.0;

    // =================================================================
    // 阶段 1：FSAL —— 若上一接受步已把 k7 缓存为 t 处的 k1，则复用
    // =================================================================
    const bool reuse_k1 = rk45_cache.has_cached_k1 && (std::abs(rk45_cache.cached_k1_t - t)<1e-15);
    // const bool reuse_k1 = false;
    if (!reuse_k1) {
        compute_dno_rhs(t, state.eta_hat, state.xi_hat, p, k1e, k1x, rk45_cache.dno);
        rk45_cache.cached_k1_t   = t;
        rk45_cache.has_cached_k1 = true;
    }

    // ---- 阶段 2 ----
    for (size_t i = 0; i < N; ++i) {
        eta_tmp[i] = state.eta_hat[i] + dt * a21 * k1e[i];
        xi_tmp[i]  = state.xi_hat[i]  + dt * a21 * k1x[i];
    }
    compute_dno_rhs(t + c2*dt, eta_tmp, xi_tmp, p, k2e, k2x, rk45_cache.dno);

    // 阶段 3 (t + c3*dt)
    for (size_t i = 0; i < N; ++i) {
        eta_tmp[i] = state.eta_hat[i] + dt * (a31 * k1e[i] + a32 * k2e[i]);
        xi_tmp[i]  = state.xi_hat[i]  + dt * (a31 * k1x[i] + a32 * k2x[i]);
    }
    compute_dno_rhs(t + c3 * dt, eta_tmp, xi_tmp, p, k3e, k3x, rk45_cache.dno);

    // 阶段 4 (t + c4*dt)
    for (size_t i = 0; i < N; ++i) {
        eta_tmp[i] = state.eta_hat[i]
                   + dt * (a41 * k1e[i] + a42 * k2e[i] + a43 * k3e[i]);
        xi_tmp[i]  = state.xi_hat[i]
                   + dt * (a41 * k1x[i] + a42 * k2x[i] + a43 * k3x[i]);
    }
    compute_dno_rhs(t + c4 * dt, eta_tmp, xi_tmp, p, k4e, k4x, rk45_cache.dno);

    // 阶段 5 (t + c5*dt)
    for (size_t i = 0; i < N; ++i) {
        eta_tmp[i] = state.eta_hat[i]
                   + dt * (a51 * k1e[i] + a52 * k2e[i]
                         + a53 * k3e[i] + a54 * k4e[i]);
        xi_tmp[i]  = state.xi_hat[i]
                   + dt * (a51 * k1x[i] + a52 * k2x[i]
                         + a53 * k3x[i] + a54 * k4x[i]);
    }
    compute_dno_rhs(t + c5 * dt, eta_tmp, xi_tmp, p, k5e, k5x, rk45_cache.dno);

    // 阶段 6 (t + dt)
    for (size_t i = 0; i < N; ++i) {
        eta_tmp[i] = state.eta_hat[i]
                   + dt * (a61 * k1e[i] + a62 * k2e[i] + a63 * k3e[i]
                         + a64 * k4e[i] + a65 * k5e[i]);
        xi_tmp[i]  = state.xi_hat[i]
                   + dt * (a61 * k1x[i] + a62 * k2x[i] + a63 * k3x[i]
                         + a64 * k4x[i] + a65 * k5x[i]);
    }
    compute_dno_rhs(t + c6 * dt, eta_tmp, xi_tmp, p, k6e, k6x, rk45_cache.dno);

    // 阶段 7 (t + dt，与 5 阶解所在点一致)
    for (size_t i = 0; i < N; ++i) {
        eta_tmp[i] = state.eta_hat[i]
                   + dt * (a71 * k1e[i] + a73 * k3e[i] + a74 * k4e[i]
                         + a75 * k5e[i] + a76 * k6e[i]);
        xi_tmp[i]  = state.xi_hat[i]
                   + dt * (a71 * k1x[i] + a73 * k3x[i] + a74 * k4x[i]
                         + a75 * k5x[i] + a76 * k6x[i]);
    }
    compute_dno_rhs(t + c7 * dt, eta_tmp, xi_tmp, p, k7e, k7x, rk45_cache.dno);

    // ---- 5 阶解(直接复用eta_tmp/xi_tmp) + 误差----
    for (size_t i = 0; i < N; ++i) {
        eta_err[i] = dt * ((b1-b1s)*k1e[i] + (b3-b3s)*k3e[i]
                         + (b4-b4s)*k4e[i] + (b5-b5s)*k5e[i]
                         + (b6-b6s)*k6e[i] + (-b7s)*k7e[i]);
        xi_err[i]  = dt * ((b1-b1s)*k1x[i] + (b3-b3s)*k3x[i]
                         + (b4-b4s)*k4x[i] + (b5-b5s)*k5x[i]
                         + (b6-b6s)*k6x[i] + (-b7s)*k7x[i]);
    }

    // ... 误差范数与接受/拒绝同原逻辑 ...
    // ---- 加权 RMS 误差范数 ----
    double sum_sq = 0.0;
    for (size_t i = 0; i < N; ++i) {
        const double se = atol + rtol * std::abs(state.eta_hat[i]);
        const double sx = atol + rtol * std::abs(state.xi_hat[i]);
        const double re = std::abs(eta_err[i]) / se;
        const double rx = std::abs(xi_err[i])  / sx;
        sum_sq += re * re + rx * rx;
    }
    const double err_norm = std::sqrt(sum_sq / (2.0 * static_cast<double>(N)));

    RK45AttemptResult result;
    result.err_norm = err_norm;

    // ---- 接受 / 拒绝 ----
    if (err_norm <= 1.0) { // accepted
        result.accepted = true;
        state.eta_hat = eta_tmp;
        state.xi_hat  = xi_tmp;

        // ------ FSAL：k7 位于 (t+dt, state_new)，作为下一步 k1 复用 ------
        // std::swap 对 vector 是 O(1)（仅交换指针）
        std::swap(k1e, k7e);
        std::swap(k1x, k7x);

        rk45_cache.cached_k1_t = t + dt;
        rk45_cache.has_cached_k1 = true;   // 显式保证（虽然之前也是 true）
    } else {
        result.accepted = false;   // state 保持不变
    }

    // ---- 建议下一步步长 ----
    // 误差 ∝ dt^(p+1)，p=4 => dt_new = dt * err^(-1/5)
    constexpr double expo = 1.0 / 5.0;
    double factor = (err_norm > 0.0)
                  ? safety * std::pow(err_norm, -expo)
                  : max_factor;
    factor = std::min(max_factor, std::max(min_factor, factor));
    result.new_dt = dt * factor;

    return result;
}

// =====================================================================
// 自适应积分主循环
// =====================================================================

size_t dno_time_integrate(
    DNOState& state, // input the initial state
    double t0, double t_end,
    double dt_init,
    double rtol, double atol,
    const DNOSystemParams& p,
    size_t max_steps,
    DNOOutputRecorder* recorder)
{
    if (dt_init <= 0.0) {
        throw std::invalid_argument(
            "dno_time_integrate: dt_init must be positive");
    }
    if (t_end <= t0) {
        throw std::invalid_argument(
            "dno_time_integrate: t_end must exceed t0");
    }
    if (state.eta_hat.empty() || state.eta_hat.size() != state.xi_hat.size()) {
        throw std::invalid_argument(
            "dno_time_integrate: state size mismatch or empty");
    }
    if (!p.bc_func) throw std::invalid_argument("dno_time_integrate: beta gamma function is not defined!(empty pointer)");

    const size_t N = state.eta_hat.size();

    // ------------------------------------------------------------------
    // 【新增】低通滤波参数配置
    // ------------------------------------------------------------------
    // 计算最大波数，用 std::max_element 自动搜索绝对值最大项
    auto k_max_it = std::max_element(
        p.k_axis.begin(), p.k_axis.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); }
    );
    if (k_max_it == p.k_axis.end()) {
        throw std::runtime_error("dno_time_integrate: k_axis is empty");
    }
    const double k_max = std::abs(*k_max_it);
    
    const double k_cut = 1.0 * k_max;      // 截断 75% 的最大波数
    const double filter_width = 1.0;        // 五点滤波器过渡带缩放
    const bool enable_filter = true;        // 启用开关
    const double filter_interval = 0.01; // 滤波间隔时长

    // 在循环外部预先计算一次传递函数 H(k)，避免每步重复计算
    const std::vector<double> H_lowpass =
        build_lowpass_transfer(p.k_axis, k_cut, LowPassType::Exponential, filter_width, 36);

    // ------------------------------------------------------------------
    // 记录时刻的 β 由 bc_func 现场组装，保持与积分器内部一致。
    // ------------------------------------------------------------------
    auto emit_record = [&](double t_rec) {
        std::vector<std::complex<double>> beta_hat(N, {0.0, 0.0});
        std::vector<std::complex<double>> gamma_hat(N, {0.0, 0.0});
        p.bc_func(t_rec, beta_hat, gamma_hat);
        recorder->record(t_rec, state.eta_hat, state.xi_hat, beta_hat, gamma_hat, p);
    };

    // 初始时刻记录
    if (recorder && recorder->enabled()) {
        emit_record(t0);
    }

    // ------------------------------------------------------------------
    // 自适应积分主循环
    // ------------------------------------------------------------------
    double t  = t0;
    double dt = dt_init;
    size_t n_steps = 0;
    double next_filter_t = t0 + filter_interval; // 下一个滤波时刻

    while (t < t_end) {
        // 限制补偿，防止越过终点时间
        if (t + dt > t_end) {
            dt = t_end - t;
        }
        // 限制步长，不要跨过下一个滤波时刻（如果开启滤波）
        if (enable_filter && (t + dt > next_filter_t)) {
            dt = next_filter_t - t;
        }

        const RK45AttemptResult res =
            rk45_step(t, dt, p, state, rtol, atol, 0.85);

        if (res.accepted) {
            t += dt;
            ++n_steps;
            if (n_steps > max_steps) {
                throw std::runtime_error(
                    "dno_time_integrate: max_steps exceeded");
            }
            // 低通滤波器
            if (enable_filter && t >= next_filter_t) {
                for (size_t i = 0; i < N; ++i) {
                    state.eta_hat[i] *= H_lowpass[i];
                    state.xi_hat[i]  *= H_lowpass[i];
                }
                // 滤波后让 FSAL 缓存失效，下一步强制重算 k1
                rk45_cache.has_cached_k1 = false;

                while (next_filter_t <= t) next_filter_t += filter_interval;
            }

            // record
            if (recorder && recorder->due(t)) {
                emit_record(t);
            }
        }

        if (n_steps % 100 == 0 || !res.accepted) {
            std::cout << "step=" << n_steps
              << "  t=" << std::scientific << std::setprecision(6) << t
              << "  dt_used=" << dt
              << "  new_dt=" << res.new_dt
              << "  err=" << res.err_norm
              << "  accepted=" << (res.accepted ? "Y" : "N")
              << std::endl;
        }

        

        dt = res.new_dt;

        if (dt < 1e-18) {
            throw std::runtime_error(
                "dno_time_integrate: dt underflow, integration stalled");
        }
    }
    return n_steps;
}
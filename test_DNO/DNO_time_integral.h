#ifndef DNO_TIME_INTEGRAL_H
#define DNO_TIME_INTEGRAL_H

#include "DNO_calc_OE.h"
#include "DNO_output_control.h"

#include <vector>
#include <complex>
#include <functional>
#include <cstddef>

// ---------------------------------------------------------------------
// DNO 状态：η、ξ 的傅里叶系数
// ---------------------------------------------------------------------
struct DNOState {
    std::vector<std::complex<double>> eta_hat;
    std::vector<std::complex<double>> xi_hat;
};

// ---------------------------------------------------------------------
// 单步尝试结果
// ---------------------------------------------------------------------
struct RK45AttemptResult {
    bool   accepted;    // 该步是否被接受
    double new_dt;      // 建议的下一步步长
    double err_norm;    // 加权 RMS 误差范数
};

// ---------------------------------------------------------------------
// RK45 阶段缓冲池：只在首次调用时分配，之后原地复用。
// ---------------------------------------------------------------------
struct RK45BufferCache
{
    std::vector<std::complex<double>> k1e, k1x;
    std::vector<std::complex<double>> k2e, k2x;
    std::vector<std::complex<double>> k3e, k3x;
    std::vector<std::complex<double>> k4e, k4x;
    std::vector<std::complex<double>> k5e, k5x;
    std::vector<std::complex<double>> k6e, k6x;
    std::vector<std::complex<double>> k7e, k7x;
    std::vector<std::complex<double>> eta_tmp, xi_tmp;
    std::vector<std::complex<double>> eta_err, xi_err;

    // ---- FSAL 缓存状态 ----
    double cached_k1_t  = 0.0;      // 缓存 k1 所在的时刻
    bool   has_cached_k1 = false;   // 缓存是否有效

    DNOBufferCache dno;

    void ensure_size(size_t N, int DNO_M, int DNO_Mb)
    {
        auto needs = [N](const std::vector<std::complex<double>>& v) {
            return v.size() != N;
        };
        if (needs(k1e) || needs(k1x) ||
            needs(k2e) || needs(k2x) ||
            needs(k3e) || needs(k3x) ||
            needs(k4e) || needs(k4x) ||
            needs(k5e) || needs(k5x) ||
            needs(k6e) || needs(k6x) ||
            needs(k7e) || needs(k7x) ||
            needs(eta_tmp) || needs(xi_tmp) ||
            needs(eta_err) || needs(xi_err))
        {
            auto alloc_N = [N](std::vector<std::complex<double>>& v){
                v.resize(N);
            };
            alloc_N(k1e); alloc_N(k1x); alloc_N(k2e); alloc_N(k2x);
            alloc_N(k3e); alloc_N(k3x); alloc_N(k4e); alloc_N(k4x);
            alloc_N(k5e); alloc_N(k5x); alloc_N(k6e); alloc_N(k6x);
            alloc_N(k7e); alloc_N(k7x);
            alloc_N(eta_tmp); alloc_N(xi_tmp);
            alloc_N(eta_err); alloc_N(xi_err);

            has_cached_k1 = false;   // ← 加这一行：重新分配后旧缓存失效
        }
        dno.ensure_size(N, DNO_M, DNO_Mb);
    }
};

// ---------------------------------------------------------------------
// Dormand-Prince RK45(5(4)) 单步
//
//   - 接受：state 被更新为 t+dt 处的 5 阶解；
//   - 拒绝：state 保持不变，调用者需以更小的 dt 重试。
//
// 参数：
//   t, dt            当前时刻与尝试步长
//   p                系统参数（含 β/γ 回调）
//   state            输入/输出：当前状态
//   rtol, atol       相对/绝对误差容限
//   safety           步长安全因子（默认 0.9）
//   min_factor       单步最小步长缩放（默认 0.2）
//   max_factor       单步最大步长缩放（默认 5.0）
// ---------------------------------------------------------------------
RK45AttemptResult rk45_step(
    double t, double dt,
    DNOState& state,
    double rtol, double atol,
    double safety     = 0.9,
    double min_factor = 0.2,
    double max_factor = 5.0);

// ---------------------------------------------------------------------
// 自适应步长积分主循环：从 t0 积分到 t_end
//
//   - 当剩余时间不足 dt 时，把 dt 收缩到恰好落在 t_end；
//   - 返回实际接受的步数；
//   - 若 dt 下溢到 阈值 以下或步数超过 max_steps，抛异常。
// ---------------------------------------------------------------------
size_t dno_time_integrate(
    DNOState& state,
    double dt_init,
    bool enable_filter, // whether to apply low-pass filtering during integration
    double filter_interval, // time interval for low-pass filtering
    double rtol, 
    double atol,
    size_t max_steps = 100000);

#endif // DNO_TIME_INTEGRAL_H
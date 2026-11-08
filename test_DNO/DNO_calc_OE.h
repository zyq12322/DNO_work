#ifndef DNO_CALC_OE_H
#define DNO_CALC_OE_H

#include "var_op_function.h"
#include "dealiaser.h"

#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <cassert>

// external initialization required
// [0!,1!,...]
extern std::vector<double> factorial_list;// [max( (DNO_M + 1) , (DNO_Mb + 1) , 2 )]
// \eta  \beta and k pow list in Fourier space.-the index corresponds to the power.
// external initialization required
// [eta^0, eta^1, ...] in Fourier space
extern std::vector<std::vector<std::complex<double>>> eta_pow_hat;//[max( (DNO_M + 1), 2 ) , len(x)]
// [beta^0, beta^1, ...] in Fourier space
extern std::vector<std::vector<std::complex<double>>> beta_pow_hat;//[max( (DNO_Mb + 1), 2 ) , len(x)]
// powers of wave number axis k , [k^0, k^1, ...] 
extern std::vector<std::vector<std::complex<double>>> D_pow_hat;//[max( (DNO_M + 1) , (DNO_Mb + 1) , 2 ) , len(x)]

// =====================================================================
// DNOBufferCache
//
// 跨 RK45 阶段/时间步复用全部中间缓冲区，避免每阶段重建阶数表。
//
// 线程安全性：非线程安全。每个独立运行 DNO 计算的线程应各自持有
// 一个 DNOBufferCache 实例。参考 rk45_step 中的 static thread_local 用法。
//
// 典型用法（时间积分器内部）：
//     static thread_local DNOBufferCache cache;
//     cache.ensure_size(N, DNO_M, DNO_Mb);
//     partial_eta_partial_t_2d(..., &cache);
//
// 独立调用（如 main.cpp 扫描）：
//     DNOBufferCache cache;
//     for (...) { ... partial_eta_partial_t_2d(..., &cache); }
// =====================================================================
struct DNOBufferCache
{
    // ---- partial_eta_partial_t_2d ----
    std::vector<std::complex<double>> G1_xi_hat;
    std::vector<std::complex<double>> G2_gamma_hat;
    std::vector<std::vector<std::complex<double>>> G1_m0_m1_list;
    std::vector<std::vector<std::complex<double>>> G2_m0_m1_list;

    // ---- partial_xi_partial_t_2d ----
    std::vector<std::complex<double>> eta_phy;
    std::vector<std::complex<double>> xi_phy;
    std::vector<std::complex<double>> G_phy;
    std::vector<std::complex<double>> eta_x_hat;
    std::vector<std::complex<double>> xi_x_hat;
    std::vector<std::complex<double>> eta_x_phy;
    std::vector<std::complex<double>> xi_x_phy;
    std::vector<std::complex<double>> eta_x_2_hat;
    std::vector<std::complex<double>> xi_x_2_hat;
    std::vector<std::complex<double>> G_2_hat;
    std::vector<std::complex<double>> eta_x_xi_x_hat;
    std::vector<std::complex<double>> eta_x_xi_x_G_hat;
    std::vector<std::complex<double>> term1_hat;
    std::vector<std::complex<double>> term1_phy;
    std::vector<std::complex<double>> term2_phy;
    std::vector<std::complex<double>> rhs_phy;

    // ---- compute_dno_rhs 使用 ----
    std::vector<std::complex<double>> beta_hat;
    std::vector<std::complex<double>> gamma_hat;

    // ---- 暂存 ----
    std::vector<std::complex<double>> scratch1;  // 例如 odd_term / even_term
    std::vector<std::complex<double>> scratch2;  // 例如 beta_phy / eta_phy

    // ---- 记录当前尺寸，保证 ensure_size 幂等 ----
    size_t cached_N  = 0;
    int    cached_M  = -1;
    int    cached_Mb = -1;

    // 若 (N, DNO_M, DNO_Mb) 与上次相同，本函数为空操作。
    void ensure_size(size_t N, int DNO_M, int DNO_Mb)
    {
        if (cached_N == N && cached_M == DNO_M && cached_Mb == DNO_Mb)
            return;

        const size_t num_orders =
            (static_cast<size_t>(DNO_Mb) + 1) *
            (static_cast<size_t>(DNO_M)  + 1);

        // N 大小向量
        auto alloc_N = [N](std::vector<std::complex<double>>& v) {
            v.resize(N);
        };
        alloc_N(G1_xi_hat);            alloc_N(G2_gamma_hat);
        alloc_N(eta_phy);              alloc_N(xi_phy);
        alloc_N(G_phy);                alloc_N(eta_x_hat);
        alloc_N(xi_x_hat);             alloc_N(eta_x_phy);
        alloc_N(xi_x_phy);             alloc_N(eta_x_2_hat);
        alloc_N(xi_x_2_hat);           alloc_N(G_2_hat);
        alloc_N(eta_x_xi_x_hat);       alloc_N(eta_x_xi_x_G_hat);
        alloc_N(term1_hat);            alloc_N(term1_phy);
        alloc_N(term2_phy);            alloc_N(rhs_phy);
        alloc_N(beta_hat);             alloc_N(gamma_hat);
        alloc_N(scratch1);             alloc_N(scratch2);

        // 阶数表：仅在 num_orders 变化时重建行
        if (G1_m0_m1_list.size() != num_orders) {
            G1_m0_m1_list.assign(num_orders,
                std::vector<std::complex<double>>(N));
            G2_m0_m1_list.assign(num_orders,
                std::vector<std::complex<double>>(N));
        } else if (G1_m0_m1_list[0].size() != N) {
            for (auto& row : G1_m0_m1_list) row.resize(N);
            for (auto& row : G2_m0_m1_list) row.resize(N);
        }

        cached_N  = N;
        cached_M  = DNO_M;
        cached_Mb = DNO_Mb;
    }
};

// make sure correct memory has been allocated for factorial_list before calling this function
// fill factorial_list with - [0!,1!,2!,...]
inline void create_factorial_list() {
    // 0! = 1
    factorial_list[0] = 1;
    for (size_t i = 1; i < factorial_list.size(); ++i) {
        factorial_list[i] = factorial_list[i-1] * i;
        //std::cout << factorial_list[i] << std::endl;
    }
}

// G(\beta, \eta)[\xi, \gamma] = \sum G_1^{(m0,m1)}\xi + \sum G_2^(m0,m1)\gamma
// calculate G_1^{(m0,0)}\xi m0 = 0,1,2,3,...,DNO_Mb ; Then fill the results into G1_m0_m1_list
void G1_m0_0_2d(
    const std::vector<std::complex<double>>& xi_hat,// \xi in Fourier Space
    std::vector<std::vector<std::complex<double>>>& G1_m0_m1_list,//对上标阶数展平的数组[G1(0,0),G1(0,1),G1(1,0),G1(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    const std::vector<std::vector<std::complex<double>>>& G2_m0_m1_list,//对上标阶数展平的数组[G2(0,0),G2(0,1),G2(1,0),G2(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    double h0,// mean depth
    int DNO_M,// max power of \eta;
    int DNO_Mb,// max power of \beta;
    bool dealiase_flag);// flag for dealiasing product-true: use dealiasing for product between two vectors;

// G(\beta, \eta)[\xi, \gamma] = \sum G_1^{(m0,m1)}\xi + \sum G_2^(m0,m1)\gamma
// calculate G_2^{(m0,0)}\gamma m0 = 0,1,2,3,...,DNO_Mb ; Then fill the results into G2_m0_m1_list
void G2_m0_0_2d(
    const std::vector<std::complex<double>>& gamma_hat,// \gamma in Fourier Space
    std::vector<std::vector<std::complex<double>>>& G2_m0_m1_list,//对上标阶数展平的数组[G2(0,0),G2(0,1),G2(1,0),G2(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    double h0,// mean depth
    int DNO_M,// max power of \eta;
    int DNO_Mb,// max power of \beta;
    bool dealiase_flag);// flag for dealiasing product-true: use dealiasing for product between two vectors;

// G(\beta, \eta)[\xi, \gamma] = \sum G_1^{(m0,m1)}\xi + \sum G_2^(m0,m1)\gamma
// calculate G_1^{(0,m1)}\xi m1 = 0,1,2,3,...,DNO_M ; Then fill the results into G1_m0_m1_list
void G1_0_m1_2d(
    const std::vector<std::complex<double>>& xi_hat,// \xi in Fourier Space
    std::vector<std::vector<std::complex<double>>>& G1_m0_m1_list,//对上标阶数展平的数组[G1(0,0),G1(0,1),G1(1,0),G1(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    double h0,// mean depth
    int DNO_M,// max power of \eta;
    int DNO_Mb,// max power of \beta;
    bool dealiase_flag);// flag for dealiasing product-true: use dealiasing for product between two vectors;

// G(\beta, \eta)[\xi, \gamma] = \sum G_1^{(m0,m1)}\xi + \sum G_2^(m0,m1)\gamma
// calculate G_2^{(0,m1)}\gamma m1 = 0,1,2,3,...,DNO_M ; Then fill the results into G2_m0_m1_list
void G2_0_m1_2d(
    const std::vector<std::complex<double>>& gamma_hat,// \gamma in Fourier Space
    const std::vector<std::vector<std::complex<double>>>& G1_m0_m1_list,//对上标阶数展平的数组[G1(0,0),G1(0,1),G1(1,0),G1(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    std::vector<std::vector<std::complex<double>>>& G2_m0_m1_list,//对上标阶数展平的数组[G2(0,0),G2(0,1),G2(1,0),G2(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    double h0,// mean depth
    int DNO_M,// max power of \eta;
    int DNO_Mb,// max power of \beta;
    bool dealiase_flag);// flag for dealiasing product-true: use dealiasing for product between two vectors;

// G(\beta, \eta)[\xi, \gamma] = \sum G_1^{(m0,m1)}\xi + \sum G_2^(m0,m1)\gamma
// calculate G_1^{(m0,m1)}\xi ,  m0 = m0; m1 = 1, 2, ..., DNO_M; Then fill the results into G1_m0_m1_list
void G1_m0_m1_2d(
    int m0, // power of beta for G1 which will be calculated
    const std::vector<std::complex<double>>& xi_hat,// \xi in Fourier Space
    std::vector<std::vector<std::complex<double>>>& G1_m0_m1_list,//对上标阶数展平的数组[G1(0,0),G1(0,1),G1(1,0),G1(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    const std::vector<std::vector<std::complex<double>>>& G2_m0_m1_list,//对上标阶数展平的数组[G2(0,0),G2(0,1),G2(1,0),G2(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    double h0,// mean depth
    int DNO_M,// max power of \eta;
    int DNO_Mb,// max power of \beta;
    bool dealiase_flag);// flag for dealiasing product-true: use dealiasing for product between two vectors;

// G(\beta, \eta)[\xi, \gamma] = \sum G_1^{(m0,m1)}\xi + \sum G_2^(m0,m1)\gamma
// calculate G_2^{(m0,m1)}\gamma ,  m0 = m0; m1 = 1, 2, ..., DNO_M; Then fill the results into G2_m0_m1_list
void G2_m0_m1_2d(
    int m0, // power of beta for G2 which will be calculated
    const std::vector<std::complex<double>>& gamma_hat,// \gamma in Fourier Space
    const std::vector<std::vector<std::complex<double>>>& G1_m0_m1_list,//对上标阶数展平的数组[G1(0,0),G1(0,1),G1(1,0),G1(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    std::vector<std::vector<std::complex<double>>>& G2_m0_m1_list,//对上标阶数展平的数组[G2(0,0),G2(0,1),G2(1,0),G2(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    double h0,// mean depth
    int DNO_M,// max power of \eta;
    int DNO_Mb,// max power of \beta;
    bool dealiase_flag);// flag for dealiasing product-true: use dealiasing for product between two vectors;

// resize eta_pow_hat/beta_pow_hat/D_pow_hat/factorial_list according to DNO_M/DNO_Mb/N
// and then fill factorial_list according to it's size.
// make sure at least two elements in them because the index 1 is always used in DNO calculation.
inline void resize_DNO_workspace(int DNO_M, int DNO_Mb, size_t N)
{
    if (DNO_M < 0 || DNO_Mb < 0) {
        throw std::invalid_argument(
            "resize_DNO_workspace: DNO_M and DNO_Mb must be non-negative.");
    }
    // calculate size according to DNO_M, DNO_Mb and N(len(x))
    // if (DNO_M == 0) Max_eta_order = 2;
    int Max_eta_order = std::max(DNO_M + 1, 2);
    // if (DNO_Mb == 0) Max_beta_order = 2;
    int Max_beta_order = std::max(DNO_Mb + 1, 2);
    int Max_D_order = std::max(Max_eta_order, Max_beta_order);

    // resize global work vector
    D_pow_hat.resize(Max_D_order);
    factorial_list.resize(Max_D_order);
    eta_pow_hat.resize(Max_eta_order);
    beta_pow_hat.resize(Max_beta_order);

    for (int i = 0; i < Max_eta_order; ++i) eta_pow_hat[i].resize(N);
    for (int i = 0; i < Max_beta_order; ++i) beta_pow_hat[i].resize(N);
    for (int i = 0; i < Max_D_order; ++i) D_pow_hat[i].resize(N);

    // fill factorial_list with factorial values [0!, 1!, ...]
    create_factorial_list();
}

// make sure resize_DNO_workspace(int DNO_M, int DNO_Mb, size_t N) has been called before calling this function
// set eta_pow_hat/beta_pow_hat with [eta^0, eta^1, ...]/[beta^0, beta^1, ...] in Fourier space
// eta_hat : eta in Fourier space
// beta_hat : beta in Fourier space
// dealiase_flag : use dealiser(true) or not for nonlinear product
inline void set_eta_beta_pow_hat(const std::vector<std::complex<double>>& eta_hat, 
                                const std::vector<std::complex<double>>& beta_hat,
                                bool dealiase_flag)
{
    // check global objects
    if (!fft_obj || !dealiaser) throw std::runtime_error("Global fft_op/Dealiaser object not initialized");
    // check eta/beta have the same size - len(x)
    if (eta_hat.size() != beta_hat.size()) throw std::runtime_error("size of eta, beta is not equal in set_eta_beta_pow_hat()!");
    // N = len(x)
    size_t N = eta_hat.size();

    // eta_pow_hat[0] will not be used so can be filled with all ones.
    std::fill(eta_pow_hat[0].begin(), eta_pow_hat[0].end(), std::complex<double>(1.0, 0.0));
    // eta^1
    eta_pow_hat[1] = eta_hat;

    std::vector<std::complex<double>> eta_phy = fft_obj->ifft(eta_hat);
    for (size_t p = 2; p < eta_pow_hat.size(); ++p) {
        if (dealiase_flag) {
            (*dealiaser)(eta_pow_hat[p-1].data(), eta_hat.data(),
                          false,   // input in Fourier space
                          false,   // output in Fourier space
                          eta_pow_hat[p].data());
        }
        else {
            std::vector<std::complex<double>> eta_pow_phy = fft_obj->ifft(eta_pow_hat[p-1]);
            
            for (size_t i = 0; i < N; ++i) {
                eta_pow_hat[p][i] = eta_pow_phy[i] * eta_phy[i];
            }
            eta_pow_hat[p] = fft_obj->fft(eta_pow_hat[p]);
        }
    }

    // eta_pow_hat[0] will not be used so can be filled with all ones.
    std::fill(beta_pow_hat[0].begin(), beta_pow_hat[0].end(), std::complex<double>(1.0, 0.0));
    // eta^1
    beta_pow_hat[1] = beta_hat;

    std::vector<std::complex<double>> beta_phy = fft_obj->ifft(beta_hat);
    for (size_t p = 2; p < beta_pow_hat.size(); ++p) {
        if (dealiase_flag) {
            (*dealiaser)(beta_pow_hat[p-1].data(), beta_hat.data(),
                          false,   // input in Fourier space
                          false,   // output in Fourier space
                          beta_pow_hat[p].data());
        }
        else {
            std::vector<std::complex<double>> beta_pow_phy = fft_obj->ifft(beta_pow_hat[p-1]);
            
            for (size_t i = 0; i < N; ++i) {
                beta_pow_hat[p][i] = beta_pow_phy[i] * beta_phy[i];
            }
            beta_pow_hat[p] = fft_obj->fft(beta_pow_hat[p]);
        }
    }
}

// make sure resize_DNO_workspace(int DNO_M, int DNO_Mb, size_t N) has been called before calling this function
// set D_pow_hat with [k^0, k^1, ...], D corresponds to operator -i\partial_x (k_axis in Fourier space)
// k_axis : wave number axis , k_axis[i] = 2.0 * M_PI * fftfreq(N, dx)[i]
inline void init_D_pow_hat_2d(const std::vector<double>& k_axis) {
    // N = len(x) = len(k_axis)
    size_t N = k_axis.size();

    // D^0 must be set with all ones.
    std::fill(D_pow_hat[0].begin(), D_pow_hat[0].end(), std::complex<double>(1.0, 0.0));
    // D^1
    D_pow_hat[1].assign(k_axis.begin(), k_axis.end());
    
    for (size_t p = 2; p < D_pow_hat.size(); ++p) {
        for (size_t i = 0; i < N; ++i) {
            D_pow_hat[p][i] = D_pow_hat[p-1][i] * k_axis[i];
        }
    }
}

/**
 * @brief 一次性初始化 DNO 计算所需的全部工作区。
 *
 * 必须在任何 G1_* / G2_* 函数调用之前执行本函数。
 *
 * 本函数动作：
 *   1. 校验全局 fft_obj、dealiaser 已创建；
 *   2. check size of input vectors
 *   3. resize_DNO_workspace(DNO_M, DNO_Mb, N);
 *   4. set D_pow_hat with [k^0, k^1, ...]t（D = -i d/dx，傅里叶空间中对应乘角波数 k）；
 *   5. set eta_pow_hat/beta_pow_hat with [eta^0, eta^1, ...]/[beta^0, beta^1, ...] in Fourier space
 *
 * 调用顺序示例（2D，x物理域长度 L，格点数 N，步长 dx = L / N）：
 * @code
 *     // 1. 先创建全局 FFT / Dealiaser 对象
 *     fft_obj   = std::make_unique<fft_op>(N);
 *     dealiaser = std::make_unique<Dealiaser>(N, 0);   // 1D
 *
 *     // 2. 构造角波数轴（rad/m）：
 *     //    fftfreq 返回空间频率 f[k] = k_int / (N*dx) = k_int / L，
 *     //    乘以 2π 得到角波数 k_axis[k] = 2π * k_int / L
 *     const double L = 2.0 * M_PI;          // 物理域长度
 *     auto freq = fftfreq(N, L / N);        // 空间频率，单位 1/m
 *     std::vector<double> k_axis(N);
 *     for (size_t i = 0; i < N; ++i) {
 *         k_axis[i] = 2.0 * M_PI * freq[i]; // 角波数，单位 rad/m
 *     }
 *
 *     // 3. 初始化工作区
 *     //    resize eta_pow_hat/beta_pow_hat/D_pow_hat/factorial_list according to DNO_M/DNO_Mb/N
 *     //    fill factorial_list/eta_pow_hat/beta_pow_hat/D_pow_hat
 *     init_DNO_workspace_2d(DNO_M, DNO_Mb, N, k_axis,
 *                        eta_hat, beta_hat, dealiase_flag);
 *
 *     // 4. 然后即可调用 G1_* / G2_* 系列 计算DNO结果
 *     // calculate DNO result
       std::vector<std::complex<double>> G_hat(N, std::complex<double>(0.0, 0.0));
       partial_eta_partial_t_2d(xi_hat, gamma_hat, G_hat, mean_depth, DNO_M, DNO_Mb, dealiase_flag);
 * @endcode
 *
 * @param DNO_M         η 的最大幂次（>= 0）
 * @param DNO_Mb        β 的最大幂次（>= 0）
 * @param N             空间格点数（与 fft_obj 的输入长度一致）
 * @param k_axis        角波数轴（rad/m），长度必须为 N。
 *                      1D 时通过 2π * fftfreq(N, dx) 构造；
 *                      2D 时按行主序展平为 Nx*Ny 的逐点波数。
 * @param eta_hat       η 的傅里叶表示，长度必须为 N
 * @param beta_hat      β 的傅里叶表示，长度必须为 N
 * @param dealiase_flag 是否使用 dealiaser 进行反混叠乘法
 *
 * @throws std::runtime_error
 *   - fft_obj 或 dealiaser 未创建
 *   - k_axis.size() != N
 *   - eta_hat.size() != N 或 beta_hat.size() != N
 * @throws std::invalid_argument
 *   - DNO_M < 0 或 DNO_Mb < 0
 *   - N == 0
 */
inline void init_DNO_workspace_2d(
    int DNO_M,
    int DNO_Mb,
    size_t N, // length of x_axis(or k_axis)
    const std::vector<double>& k_axis,
    const std::vector<std::complex<double>>& eta_hat,
    const std::vector<std::complex<double>>& beta_hat,
    bool dealiase_flag)
{
    // ---- 1. 参数校验 ----
    if (!fft_obj) {
        throw std::runtime_error(
            "init_DNO_workspace: fft_obj is not initialized. "
            "Please create fft_obj first.");
    }
    if (!dealiaser) {
        throw std::runtime_error(
            "init_DNO_workspace: dealiaser is not initialized. "
            "Please create dealiaser first.");
    }
    if (N == 0) {
        throw std::invalid_argument(
            "init_DNO_workspace: N must be positive.");
    }
    if (DNO_M < 0 || DNO_Mb < 0) {
        throw std::invalid_argument(
            "init_DNO_workspace: DNO_M and DNO_Mb must be non-negative.");
    }
    if (k_axis.size() != N) {
        throw std::runtime_error(
            "init_DNO_workspace: k_axis size does not match N.");
    }
    if (eta_hat.size() != N || beta_hat.size() != N) {
        throw std::runtime_error(
            "init_DNO_workspace: eta_hat or beta_hat size does not match N.");
    }

    // ---- 尺寸检查：若四个全局工作区的尺寸已经与目标一致，则跳过 resize ----
    // 目标尺寸需与 resize_DNO_workspace 内部的推导完全一致
    const int Max_eta_order  = std::max(DNO_M  + 1, 2);
    const int Max_beta_order = std::max(DNO_Mb + 1, 2);
    const int Max_D_order    = std::max(Max_eta_order, Max_beta_order);

    // 检查“外层行数 + 每行长度”是否全部匹配
    auto size_ok = [N](const std::vector<std::vector<std::complex<double>>>& v,
                       int expected_outer) -> bool
    {
        if (static_cast<int>(v.size()) != expected_outer) return false;
        for (const auto& row : v) {
            if (row.size() != N) return false;
        }
        return true;
    };

    const bool workspace_size_ok =
           size_ok(eta_pow_hat,  Max_eta_order)
        && size_ok(beta_pow_hat, Max_beta_order)
        && size_ok(D_pow_hat,    Max_D_order)
        && (factorial_list.size() == static_cast<size_t>(Max_D_order));

    if (!workspace_size_ok) {
        // resize eta_pow_hat/beta_pow_hat/D_pow_hat/factorial_list according to changed DNO_M/DNO_Mb/N
        // and then fill factorial_list according to it's size.
        resize_DNO_workspace(DNO_M, DNO_Mb, N);
    }

    // set D_pow_hat with [k^0, k^1, ...]
    //     约定：D = -i d/dx，傅里叶空间中对应乘实数角波数 k
    init_D_pow_hat_2d(k_axis);

    // set eta_pow_hat/beta_pow_hat with [eta^0, eta^1, ...]/[beta^0, beta^1, ...] in Fourier space
    set_eta_beta_pow_hat(
        eta_hat,
        beta_hat,
        dealiase_flag);
}

// G(\beta, \eta)[\xi, \gamma] = \sum G_1^{(m0,m1)}\xi + \sum G_2^(m0,m1)\gamma
// calculate \sum G_1^{(m0,m1)}\xi ,  m0 = 0, 1, ... , DNO_Mb; m1 = 0, 1, ... , DNO_M; Then fill the results into G1_xi_hat
void G1_2d(
    const std::vector<std::complex<double>>& xi_hat,// \xi in Fourier Space
    std::vector<std::complex<double>>& G1_xi_hat,// store results of \sum G_1^{(m0,m1)}\xi
    std::vector<std::vector<std::complex<double>>>& G1_m0_m1_list,//对上标阶数展平的数组[G1(0,0),G1(0,1),G1(1,0),G1(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    std::vector<std::vector<std::complex<double>>>& G2_m0_m1_list,//对上标阶数展平的数组[G2(0,0),G2(0,1),G2(1,0),G2(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    double h0,// mean depth
    int DNO_M,// max power of \eta;
    int DNO_Mb,// max power of \beta;
    bool dealiase_flag);// flag for dealiasing product-true: use dealiasing for product between two vectors;

// G(\beta, \eta)[\xi, \gamma] = \sum G_1^{(m0,m1)}\xi + \sum G_2^(m0,m1)\gamma
// calculate \sum G_2^{(m0,m1)}\gamma ,  m0 = 0, 1, ... , DNO_Mb; m1 = 0, 1, ... , DNO_M; Then fill the results into G2_gamma_hat
void G2_2d(
    const std::vector<std::complex<double>>& gamma_hat,// \gamma in Fourier Space
    std::vector<std::complex<double>>& G2_gamma_hat,// store results of \sum G_1^{(m0,m1)}\xi
    std::vector<std::vector<std::complex<double>>>& G1_m0_m1_list,//对上标阶数展平的数组[G1(0,0),G1(0,1),G1(1,0),G1(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    std::vector<std::vector<std::complex<double>>>& G2_m0_m1_list,//对上标阶数展平的数组[G2(0,0),G2(0,1),G2(1,0),G2(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    double h0,// mean depth
    int DNO_M,// max power of \eta;
    int DNO_Mb,// max power of \beta;
    bool dealiase_flag);// flag for dealiasing product-true: use dealiasing for product between two vectors;


// \partial_eta_partial_t = G(\beta, \eta)[\xi, \gamma] = \sum G_1^{(m0,m1)}\xi + \sum G_2^(m0,m1)\gamma
// \gamma = \partial_t \beta is known
// make sure eta_pow_hat/beta_pow_hat/D_pow_hat/factorial_list has been refreshed before call this function.
void partial_eta_partial_t_2d(
    const std::vector<std::complex<double>>& xi_hat,
    const std::vector<std::complex<double>>& gamma_hat,
    std::vector<std::complex<double>>& eta_t_hat,
    double h0, int DNO_M, int DNO_Mb, bool dealiase_flag,
    DNOBufferCache* cache = nullptr);

// \partial_xi_partial_t = -g \eta - 0.5 * (\xi_x^2 - G^2 - 2 \eta_x \xi_x G) / (1 + \eta_x^2)
// make sure eta_pow_hat/beta_pow_hat/D_pow_hat/factorial_list has been refreshed before call this function.
void partial_xi_partial_t_2d(
    const std::vector<std::complex<double>>& xi_hat,
    const std::vector<std::complex<double>>& G_hat,
    std::vector<std::complex<double>>& xi_t_hat,
    double g, int DNO_M, int DNO_Mb, bool dealiase_flag,
    DNOBufferCache* cache = nullptr);

#endif
#include "var_op_function.h"
#include <cmath>

// global fft_op object - external initialization required
std::unique_ptr<fft_op> fft_obj;

/**
 * @brief Computes L2 error metrics between two real-valued vectors.
 *
 * Calculates:
 * - RMSE (root mean square error)
 * - Relative L2 error (normalized by signal energy)
 *
 * @param orig Original signal vector.
 * @param rec  Reconstructed signal vector.
 * @return Pair {RMSE, relative L2 error}.
 *
 * @throws std::runtime_error if vector sizes differ.
 */
std::pair<double, double> compute_l2_error(const std::vector<double>& orig, const std::vector<double>& rec) {
    if (orig.size() != rec.size()) {
        throw std::runtime_error("Two vectors must have same size in compute_l2_error!");
    }
    double sum_sq_diff = 0.0;
    double sum_sq_orig = 0.0;
    size_t n = orig.size();
    if (n == 0) return {0.0, 0.0};
    for (size_t i = 0; i < n; ++i) {
        double diff = orig[i] - rec[i];
        sum_sq_diff += diff * diff;
        sum_sq_orig += orig[i] * orig[i];
    }
    double rmse = std::sqrt(sum_sq_diff / n);
    double rel_err = (sum_sq_orig > 0.0) ? std::sqrt(sum_sq_diff / sum_sq_orig) : 0.0;
    return std::make_pair(rmse, rel_err);
}

/**
 * @brief Computes L2 error metrics between two complex-valued vectors.
 *
 * Calculates:
 * - RMSE based on complex magnitude difference
 * - Relative L2 error normalized by original signal energy
 *
 * @param orig Original complex signal vector.
 * @param rec  Reconstructed complex signal vector.
 * @return Pair {RMSE, relative L2 error}.
 *
 * @throws std::runtime_error if vector sizes differ.
 */
std::pair<double, double> compute_l2_error(const std::vector<std::complex<double>>& orig, const std::vector<std::complex<double>>& rec) {
    if (orig.size() != rec.size()) {
        throw std::runtime_error("Two vectors must have same size in compute_l2_error!");
    }
    
    double sum_sq_diff = 0.0;
    double sum_sq_orig = 0.0;
    size_t n = orig.size();
    if (n == 0) return {0.0, 0.0};
    for (size_t i = 0; i < n; ++i) {
        double diff = std::abs(orig[i] - rec[i]);
        sum_sq_diff += diff * diff;
        sum_sq_orig += std::norm(orig[i]);
    }
    double rmse = std::sqrt(sum_sq_diff / n);
    double rel_err = (sum_sq_orig > 0.0) ? std::sqrt(sum_sq_diff / sum_sq_orig) : 0.0;
    return std::make_pair(rmse, rel_err);
}

// ---------------------------------------------------------------------
// Implementation 1D fft tools.
// ---------------------------------------------------------------------

//class fft1d::constructor
fft1d::fft1d(size_t N)
{
    if (N == 0)
    {
        throw std::invalid_argument(
            "fft1d: N must be positive."
        );
    }
    N_ = N;
    in_buffer_.resize(N_);
    out_buffer_.resize(N_);

    forward_plan_ =
        fftw_plan_dft_1d(
            N_,
            reinterpret_cast<fftw_complex*>(
                in_buffer_.data()),
            reinterpret_cast<fftw_complex*>(
                out_buffer_.data()),
            FFTW_FORWARD,
            FFTW_MEASURE
        );

    if (!forward_plan_)
    {
        throw std::runtime_error(
            "fft1d: failed to create forward plan."
        );
    }

    backward_plan_ =
        fftw_plan_dft_1d(
            N_,
            reinterpret_cast<fftw_complex*>(
                in_buffer_.data()),
            reinterpret_cast<fftw_complex*>(
                out_buffer_.data()),
            FFTW_BACKWARD,
            FFTW_MEASURE
        );

    if (!backward_plan_)
    {
        fftw_destroy_plan(
            forward_plan_);

        throw std::runtime_error(
            "fft1d: failed to create inverse plan."
        );
    }
}
//class fft1d::destructor
fft1d::~fft1d()
{
    fftw_destroy_plan(
        forward_plan_);

    fftw_destroy_plan(
        backward_plan_);
}
// forward transform
const std::vector<std::complex<double>>&
fft1d::fft(
    const std::vector<std::complex<double>>& input)
{
    if (input.size() != N_)
    {
        throw std::invalid_argument(
            "fft1d::fft: wrong input size."
        );
    }

    std::copy(
        input.begin(),
        input.end(),
        in_buffer_.begin()
    );

    fftw_execute(
        forward_plan_);

    return out_buffer_;
}
// backward transform
const std::vector<std::complex<double>>&
fft1d::ifft(
    const std::vector<std::complex<double>>& input)
{
    if (input.size() != N_)
    {
        throw std::invalid_argument(
            "fft1d::ifft: wrong input size."
        );
    }

    std::copy(
        input.begin(),
        input.end(),
        in_buffer_.begin()
    );

    fftw_execute(
        backward_plan_);

    const double scale =
        1.0 / static_cast<double>(N_);

    for (auto& z : out_buffer_)
    {
        z *= scale;
    }

    return out_buffer_;
}

// ---------------------------------------------------------------------
// Implementation 2D fft tools.
// ---------------------------------------------------------------------

fft2d::fft2d(size_t Nx, size_t Ny)
    : Nx_(Nx), Ny_(Ny), total_(Nx * Ny)
{
    if (Nx == 0 || Ny == 0) {
        throw std::invalid_argument("fft2d: Nx and Ny must be positive.");
    }

    in_buffer_.resize(total_);
    out_buffer_.resize(total_);

    // Create forward plan (FFTW_FORWARD)
    int n[2] = { static_cast<int>(Nx_), static_cast<int>(Ny_) };
    forward_plan_ = fftw_plan_dft_2d(
        n[0], n[1],
        reinterpret_cast<fftw_complex*>(in_buffer_.data()),
        reinterpret_cast<fftw_complex*>(out_buffer_.data()),
        FFTW_FORWARD,
        FFTW_MEASURE
    );
    if (!forward_plan_) {
        throw std::runtime_error("fft2d: failed to create forward plan.");
    }

    // Create backward plan (FFTW_BACKWARD)
    backward_plan_ = fftw_plan_dft_2d(
        n[0], n[1],
        reinterpret_cast<fftw_complex*>(in_buffer_.data()),
        reinterpret_cast<fftw_complex*>(out_buffer_.data()),
        FFTW_BACKWARD,
        FFTW_MEASURE
    );
    if (!backward_plan_) {
        fftw_destroy_plan(forward_plan_);
        throw std::runtime_error("fft2d: failed to create inverse plan.");
    }
}

fft2d::~fft2d()
{
    fftw_destroy_plan(forward_plan_);
    fftw_destroy_plan(backward_plan_);
}

const std::vector<std::complex<double>>&
fft2d::fft2(const std::vector<std::complex<double>>& input)
{
    if (input.size() != total_) {
        throw std::invalid_argument("fft2d::fft2: wrong input size.");
    }

    std::copy(input.begin(), input.end(), in_buffer_.begin());
    fftw_execute(forward_plan_);
    return out_buffer_;
}

const std::vector<std::complex<double>>&
fft2d::ifft2(const std::vector<std::complex<double>>& input)
{
    if (input.size() != total_) {
        throw std::invalid_argument("fft2d::ifft2: wrong input size.");
    }

    std::copy(input.begin(), input.end(), in_buffer_.begin());
    fftw_execute(backward_plan_);

    const double scale = 1.0 / static_cast<double>(total_);
    for (auto& z : out_buffer_) {
        z *= scale;
    }
    return out_buffer_;
}

// =====================================================================
// Low-pass filter implementation
// =====================================================================

std::vector<double> build_lowpass_transfer(
    const std::vector<double>& k_axis,
    double k_cut,
    LowPassType type,
    double width,
    double order)
{
    if (k_axis.empty())
    {
        throw std::invalid_argument(
            "build_lowpass_transfer: k_axis is empty.");
    }
    if (k_cut <= 0.0)
    {
        throw std::invalid_argument(
            "build_lowpass_transfer: k_cut must be positive.");
    }
    if (width <= 0.0)
    {
        throw std::invalid_argument(
            "build_lowpass_transfer: width must be positive.");
    }
    if (order <= 0.0)
    {
        throw std::invalid_argument(
            "build_lowpass_transfer: order must be positive.");
    }

    std::vector<double> H;
    H.reserve(k_axis.size());

    switch (type)
    {
    // -------------------------------------------------------------
    // Ideal low-pass: hard cut at k_cut.
    // -------------------------------------------------------------
    case LowPassType::Ideal:
    {
        for (double k : k_axis)
        {
            H.push_back((std::abs(k) <= k_cut) ? 1.0 : 0.0);
        }
        break;
    }

    // -------------------------------------------------------------
    // Exponential: H(k) = exp(-order * (|k|/k_cut)^order)
    //
    // - order = 2  -> Gaussian-like
    // - order -> 1 -> exponential decay
    // - order -> ∞ -> approaches ideal filter
    // width rescales k_cut (k_eff = k_cut * width).
    // -------------------------------------------------------------
    case LowPassType::Exponential:
    {
        const double k_eff = k_cut * width;
        for (double k : k_axis)
        {
            const double r = std::abs(k) / k_eff;
            H.push_back(std::exp(-order * std::pow(r, order)));
        }
        break;
    }

    // -------------------------------------------------------------
    // Gaussian: H(k) = exp(-0.5 * (|k|/sigma)^2)
    // sigma = k_cut * width.
    // -------------------------------------------------------------
    case LowPassType::Gaussian:
    {
        const double sigma = k_cut * width;
        for (double k : k_axis)
        {
            const double r = std::abs(k) / sigma;
            H.push_back(std::exp(-0.5 * r * r));
        }
        break;
    }
    // -------------------------------------------------------------
    // Five-point smooth taper (基于给定的余弦公式):
    //
    //   A(k) = 1/8 * [ 5 + 4*cos(pi*|k|/k_cut) - cos(2*pi*|k|/k_cut) ]
    //
    // - A(0) = 1
    // - A(k_cut) = 0
    // - 当 |k| > k_cut 时，强制截断为 0，防止高频震荡反弹。
    // 
    // 而是直接使用 k_cut 作为公式中的 k_N。
    // -------------------------------------------------------------
    case LowPassType::FivePoint:
    {
        const double sigma = k_cut * width;
        for (double k : k_axis)
        {
            const double ak = std::abs(k);

            if (ak <= sigma)
            {
                // 归一化波数 r = |k| / sigma，范围 [0, 1]
                const double r = ak / sigma;
                const double val = (5.0 + 4.0 * std::cos(M_PI * r) - std::cos(2.0 * M_PI * r)) / 8.0;
                
                // 避免极小值溢出（虽然该公式在 [0,1] 内恒 >= 0）
                H.push_back(std::max(0.0, val));
            }
            else
            {
                // 超过截止波数的部分完全滤除
                H.push_back(0.0);
            }
        }
        break;
    }

    // -------------------------------------------------------------
    // Butterworth: H(k) = 1 / sqrt(1 + (|k|/k_cut)^(2*order))
    // -------------------------------------------------------------
    case LowPassType::Butterworth:
    {
        for (double k : k_axis)
        {
            const double r = std::abs(k) / k_cut;
            H.push_back(1.0 / std::sqrt(1.0 + std::pow(r, 2.0 * order)));
        }
        break;
    }
    }

    return H;
}

std::vector<std::complex<double>> lowpass_filter_spectrum(
    const std::vector<double>& k_axis,
    const std::vector<std::complex<double>>& f_hat,
    double k_cut,
    LowPassType type,
    double width,
    double order)
{
    if (k_axis.size() != f_hat.size())
    {
        throw std::runtime_error(
            "lowpass_filter_spectrum: k_axis and f_hat must have the same size.");
    }

    const std::vector<double> H =
        build_lowpass_transfer(k_axis, k_cut, type, width, order);

    std::vector<std::complex<double>> out(f_hat.size());

    for (std::size_t i = 0; i < f_hat.size(); ++i)
    {
        out[i] = H[i] * f_hat[i];
    }

    return out;
}


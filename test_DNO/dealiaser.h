// dealiaser.h
#ifndef DEALIASER_H
#define DEALIASER_H

#include <fftw3.h>
#include <complex>
#include <vector>
#include <memory>
#include <cstddef>
#include <variant>
#include <stdexcept>
#include <type_traits>



class Dealiaser1D {
public:
    // N must be even
    Dealiaser1D(size_t N, unsigned planner_effort = FFTW_MEASURE);
    ~Dealiaser1D();

    // Disable copy and assignment
    Dealiaser1D(const Dealiaser1D&) = delete;
    Dealiaser1D& operator=(const Dealiaser1D&) = delete;
    Dealiaser1D(Dealiaser1D&&) = delete;
    Dealiaser1D& operator=(Dealiaser1D&&) = delete;
    

    // Main interface:
    // a, b: input arrays (physical or Fourier according to input_physical)
    // input_physical: false -> a, b are in Fourier space; true -> in physical space
    // output_physical: false -> result in Fourier space; true -> in physical space
    // result: output array (must be pre-allocated, length N)
    void operator()(const std::complex<double>* a,
                    const std::complex<double>* b,
                    bool input_physical,
                    bool output_physical,
                    std::complex<double>* result);

private:
    size_t N;          // original length (must be even)
    size_t K;          // padded length = 3*N/2
    size_t half;       // N/2
    size_t k_half;     // K/2 (unused, but kept for symmetry)
    double scale;   // scale factor = K / N = 1.5

    // Work arrays of original size - N
    std::complex<double>* work_orig;
    std::complex<double>* a_hat;
    std::complex<double>* b_hat;
    std::complex<double>* result_hat;

    // Work arrays of padded size - K
    std::complex<double>* a_hat_pad;
    std::complex<double>* b_hat_pad;
    std::complex<double>* a_pad;
    std::complex<double>* b_pad;
    std::complex<double>* prod_pad;
    std::complex<double>* prod_hat_pad;

    // FFTW plans
    fftw_plan fwd_orig_a;   // forward FFT for a (original size)
    fftw_plan fwd_orig_b;   // forward FFT for b (original size)
    fftw_plan inv_orig;     // inverse FFT for result (original size)
    fftw_plan inv_pad_a;    // inverse FFT for padded a_hat -> a_pad
    fftw_plan inv_pad_b;    // inverse FFT for padded b_hat -> b_pad
    fftw_plan fwd_pad;      // forward FFT for product in padded physical space

    // Helper functions
    void pad_and_copy(const std::complex<double>* src, std::complex<double>* dst);
    void truncate_and_copy(const std::complex<double>* src, std::complex<double>* dst);
};

class Dealiaser2D {
public:
    // Nx, Ny must be even
    Dealiaser2D(size_t Nx, size_t Ny, unsigned planner_effort = FFTW_MEASURE);
    ~Dealiaser2D();

    Dealiaser2D(const Dealiaser2D&) = delete;
    Dealiaser2D& operator=(const Dealiaser2D&) = delete;
    Dealiaser2D(Dealiaser2D&&) = delete;
    Dealiaser2D& operator=(Dealiaser2D&&) = delete;

    // Main interface:
    // a, b: input arrays (physical or Fourier, flattened row‑major)
    // input_physical: false -> a,b are in Fourier space; true -> physical space
    // output_physical: false -> result in Fourier space; true -> physical space
    // result: output array (pre‑allocated, size Nx*Ny)
    void operator()(const std::complex<double>* a,
                    const std::complex<double>* b,
                    bool input_physical,
                    bool output_physical,
                    std::complex<double>* result);

private:
    size_t Nx, Ny;          // original dimensions (even)
    size_t Kx, Ky;          // padded dimensions = 3*Nx/2, 3*Ny/2
    size_t hx, hy;          // Nx/2, Ny/2
    double scale;           // (Kx/Nx) * (Ky/Ny)

    // Original‑size arrays (Nx*Ny)
    std::complex<double>* work_orig;
    std::complex<double>* a_hat;
    std::complex<double>* b_hat;
    std::complex<double>* result_hat;

    // Padded‑size arrays (Kx*Ky)
    std::complex<double>* a_hat_pad;
    std::complex<double>* b_hat_pad;
    std::complex<double>* a_pad;
    std::complex<double>* b_pad;
    std::complex<double>* prod_pad;
    std::complex<double>* prod_hat_pad;

    // FFTW plans
    fftw_plan fwd_orig_a;    // forward FFT for a (original size)
    fftw_plan fwd_orig_b;    // forward FFT for b (original size)
    fftw_plan inv_orig;      // inverse FFT for result (original size)
    fftw_plan inv_pad_a;     // inverse FFT for padded a_hat -> a_pad
    fftw_plan inv_pad_b;     // inverse FFT for padded b_hat -> b_pad
    fftw_plan fwd_pad;       // forward FFT for product (padded size)

    // Helper functions
    void pad_and_copy(const std::complex<double>* src, std::complex<double>* dst);
    void truncate_and_copy(const std::complex<double>* src, std::complex<double>* dst);
};

class Dealiaser {
public:
    // 统一构造函数：Ny == 0 表示一维，否则二维
    Dealiaser(size_t Nx, size_t Ny, unsigned planner_effort = FFTW_MEASURE)
        : dim(Ny == 0 ? 1 : 2)
    {
        if (Ny == 0) impl.emplace<Dealiaser1D>(Nx, planner_effort);
        else         impl.emplace<Dealiaser2D>(Nx, Ny, planner_effort);
    }


    // Disable copy and assignment (since variant holds non-copyable types if we delete copy)
    Dealiaser(const Dealiaser&) = delete;
    Dealiaser& operator=(const Dealiaser&) = delete;

    // Move is allowed if the underlying types are movable (they are not, because of fftw_plan)
    // But we can implement move manually if needed; for simplicity, delete.
    Dealiaser(Dealiaser&&) = delete;
    Dealiaser& operator=(Dealiaser&&) = delete;

    // Main interface: forwards to the stored dealiaser
    void operator()(const std::complex<double>* a,
                    const std::complex<double>* b,
                    bool input_physical,
                    bool output_physical,
                    std::complex<double>* result) {
        std::visit([&](auto& dealiaser) {
            using T = std::decay_t<decltype(dealiaser)>;
            if constexpr (!std::is_same_v<T, std::monostate>)
                dealiaser(a, b, input_physical, output_physical, result);
            else throw std::logic_error("Dealiaser not initialized");
        }, impl);
    }

    // Optional: query dimension
    int dimension() const noexcept { return dim; }

private:
    int dim;
    std::variant<std::monostate, Dealiaser1D, Dealiaser2D> impl;
};

// global object - external initialization required
extern std::unique_ptr<Dealiaser> dealiaser;

#endif // DEALIASER_H
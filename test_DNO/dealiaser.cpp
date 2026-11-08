// dealiaser.cpp
#include "dealiaser.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <cassert>
#include <cstdlib>   // for std::exit
#include <stdexcept>
// global object - external initialization required
std::unique_ptr<Dealiaser> dealiaser;

Dealiaser1D::Dealiaser1D(size_t N, unsigned planner_effort)
    : N(N), K(3*N/2), half(N/2), k_half(K/2), scale(double(K)/double(N))
{
    if (N % 2 != 0) {
        throw std::runtime_error("Dealiaser1D: N must be even");
    }

    // Allocate memory (set all pointers to nullptr first)
    work_orig = a_hat = b_hat = result_hat = nullptr;
    a_hat_pad = b_hat_pad = a_pad = b_pad = prod_pad = prod_hat_pad = nullptr;
    fwd_orig_a = fwd_orig_b = inv_orig = inv_pad_a = inv_pad_b = fwd_pad = nullptr;

    auto allocate = [&](auto& ptr, size_t cnt, const char* name) {
        ptr = reinterpret_cast<std::complex<double>*>(fftw_malloc(cnt * sizeof(fftw_complex)));
        if (!ptr) {
            throw std::runtime_error(std::string("fftw_malloc failed for ") + name);
        }
    };

    try {
        allocate(work_orig,   N, "work_orig");
        allocate(a_hat,       N, "a_hat");
        allocate(b_hat,       N, "b_hat");
        allocate(result_hat,  N, "result_hat");
        allocate(a_hat_pad,   K, "a_hat_pad");
        allocate(b_hat_pad,   K, "b_hat_pad");
        allocate(a_pad,       K, "a_pad");
        allocate(b_pad,       K, "b_pad");
        allocate(prod_pad,    K, "prod_pad");
        allocate(prod_hat_pad, K, "prod_hat_pad");
    } catch (const std::runtime_error&) {
        // Free all already allocated memory (nullptr is safe)
        fftw_free(work_orig);   fftw_free(a_hat);
        fftw_free(b_hat);       fftw_free(result_hat);
        fftw_free(a_hat_pad);
        fftw_free(b_hat_pad);   fftw_free(a_pad);
        fftw_free(b_pad);       fftw_free(prod_pad);
        fftw_free(prod_hat_pad);
        throw;
    }

    

    // Create plans (if any fails, clean up already created plans and memory)
    try {
        // Original size plans (all using N)
        fwd_orig_a = fftw_plan_dft_1d(static_cast<int>(N),
                                      reinterpret_cast<fftw_complex*>(work_orig),
                                      reinterpret_cast<fftw_complex*>(a_hat),
                                      FFTW_FORWARD, planner_effort);
        if (!fwd_orig_a) throw std::runtime_error("fwd_orig_a plan failed");

        fwd_orig_b = fftw_plan_dft_1d(static_cast<int>(N),
                                      reinterpret_cast<fftw_complex*>(work_orig),
                                      reinterpret_cast<fftw_complex*>(b_hat),
                                      FFTW_FORWARD, planner_effort);
        if (!fwd_orig_b) throw std::runtime_error("fwd_orig_b plan failed");

        inv_orig = fftw_plan_dft_1d(static_cast<int>(N),
                                    reinterpret_cast<fftw_complex*>(result_hat),
                                    reinterpret_cast<fftw_complex*>(work_orig),
                                    FFTW_BACKWARD, planner_effort);
        if (!inv_orig) throw std::runtime_error("inv_orig plan failed");

        // Padded size plans (using K)
        inv_pad_a = fftw_plan_dft_1d(static_cast<int>(K),
                                     reinterpret_cast<fftw_complex*>(a_hat_pad),
                                     reinterpret_cast<fftw_complex*>(a_pad),
                                     FFTW_BACKWARD, planner_effort);
        if (!inv_pad_a) throw std::runtime_error("inv_pad_a plan failed");

        inv_pad_b = fftw_plan_dft_1d(static_cast<int>(K),
                                     reinterpret_cast<fftw_complex*>(b_hat_pad),
                                     reinterpret_cast<fftw_complex*>(b_pad),
                                     FFTW_BACKWARD, planner_effort);
        if (!inv_pad_b) throw std::runtime_error("inv_pad_b plan failed");

        fwd_pad = fftw_plan_dft_1d(static_cast<int>(K),
                                   reinterpret_cast<fftw_complex*>(prod_pad),
                                   reinterpret_cast<fftw_complex*>(prod_hat_pad),
                                   FFTW_FORWARD, planner_effort);
        if (!fwd_pad) throw std::runtime_error("fwd_pad plan failed");

    } catch (const std::runtime_error& e) {
        // Destroy any plans that were successfully created
        if (fwd_orig_a) fftw_destroy_plan(fwd_orig_a);
        if (fwd_orig_b) fftw_destroy_plan(fwd_orig_b);
        if (inv_orig)   fftw_destroy_plan(inv_orig);
        if (inv_pad_a)  fftw_destroy_plan(inv_pad_a);
        if (inv_pad_b)  fftw_destroy_plan(inv_pad_b);
        if (fwd_pad)    fftw_destroy_plan(fwd_pad);

        // Free all memory
        fftw_free(work_orig);   fftw_free(a_hat);
        fftw_free(b_hat);       fftw_free(result_hat);
        fftw_free(a_hat_pad);
        fftw_free(b_hat_pad);   fftw_free(a_pad);
        fftw_free(b_pad);       fftw_free(prod_pad);
        fftw_free(prod_hat_pad);

        // Reset pointers to avoid double free in destructor (which won't run anyway,
        // but good practice if we ever rethrow and the object is partially constructed)
        work_orig = a_hat = b_hat = result_hat = nullptr;
        a_hat_pad = b_hat_pad = a_pad = b_pad = prod_pad = prod_hat_pad = nullptr;
        fwd_orig_a = fwd_orig_b = inv_orig = inv_pad_a = inv_pad_b = fwd_pad = nullptr;
        // Re-throw
        throw;
    }
}

Dealiaser1D::~Dealiaser1D() {
    // Destroy plans
    fftw_destroy_plan(fwd_orig_a);
    fftw_destroy_plan(fwd_orig_b);
    fftw_destroy_plan(inv_orig);
    fftw_destroy_plan(inv_pad_a);
    fftw_destroy_plan(inv_pad_b);
    fftw_destroy_plan(fwd_pad);

    // Free memory
    fftw_free(work_orig);
    fftw_free(a_hat);
    fftw_free(b_hat);
    fftw_free(result_hat);

    fftw_free(a_hat_pad);
    fftw_free(b_hat_pad);
    fftw_free(a_pad);
    fftw_free(b_pad);
    fftw_free(prod_pad);
    fftw_free(prod_hat_pad);
}

void Dealiaser1D::pad_and_copy(const std::complex<double>* src, std::complex<double>* dst) {
    // Zero out the destination padded array
    std::fill(dst, dst + K, std::complex<double>(0.0, 0.0));
    // Copy low-frequency part to the beginning
    std::copy(src, src + half, dst);
    // Copy high-frequency part to the end
    std::copy(src + half, src + N, dst + K - half);
}

void Dealiaser1D::truncate_and_copy(const std::complex<double>* src, std::complex<double>* dst) {
    // Zero out the destination original-sized array
    std::fill(dst, dst + N, std::complex<double>(0.0, 0.0));
    // Copy low-frequency part from the beginning
    std::copy(src, src + half, dst);
    // Copy high-frequency part from the end
    std::copy(src + K - half, src + K, dst + half);
}

void Dealiaser1D::operator()(const std::complex<double>* a,//N
                             const std::complex<double>* b,//N
                             bool input_physical,
                             bool output_physical,
                             std::complex<double>* result)//N 
{
    // Step 1: Obtain Fourier representations of a and b
    // and copy them to a_hat b_hat respectively.
    // Rules: if input_physical, input in physical space;
    if (input_physical) {
        // Input is in physical space -> forward FFT
        std::copy(a, a + N, work_orig);
        fftw_execute(fwd_orig_a);   // work_orig -> a_hat
        std::copy(b, b + N, work_orig);
        fftw_execute(fwd_orig_b);   // work_orig -> b_hat
    } else {
        // Input is already in Fourier space(len == N)
        std::copy(a, a + N, a_hat);
        std::copy(b, b + N, b_hat);
    }

    // Step 2: Pad the spectra to the extended size (3N/2)
    pad_and_copy(a_hat, a_hat_pad);
    pad_and_copy(b_hat, b_hat_pad);

    // Step 3: Inverse FFT to physical space on padded grid
    fftw_execute(inv_pad_a);   // a_hat_pad -> a_pad
    fftw_execute(inv_pad_b);   // b_hat_pad -> b_pad

    // Step 4: Pointwise product in physical space
    for (size_t i = 0; i < K; ++i) {
        prod_pad[i] = a_pad[i] / double(N) * b_pad[i] / double(N);
    }

    // Step 5: Forward FFT back to Fourier space on padded grid
    fftw_execute(fwd_pad);      // prod_pad -> prod_hat_pad

    // Step 6: Truncate to original size and apply scaling
    truncate_and_copy(prod_hat_pad, result_hat);
    for (size_t i = 0; i < N; ++i) {
        result_hat[i] /= scale;
    }

    // Step 7: Output according to output_physical flag
    if (!output_physical) {
        // Output in Fourier space
        std::copy(result_hat, result_hat + N, result);
    } else {
        // Inverse FFT to physical space
        fftw_execute(inv_orig);   // result_hat -> work_orig (inverse transform)
        double inv_norm = 1.0 / double(N);
        for (size_t i = 0; i < N; ++i) {
            result[i] = work_orig[i] * inv_norm;
        }
    }
}

Dealiaser2D::Dealiaser2D(size_t Nx, size_t Ny, unsigned planner_effort)
    : Nx(Nx), Ny(Ny),
      Kx(3 * Nx / 2), Ky(3 * Ny / 2),
      hx(Nx / 2), hy(Ny / 2),
      scale(double(Kx) / double(Nx) * double(Ky) / double(Ny))
{
    if ((Nx % 2 != 0) || (Ny % 2 != 0)) {
        throw std::runtime_error("Dealiaser2D: Nx and Ny must be even");
    }

    // Initialize pointers to nullptr for safe cleanup
    work_orig = a_hat = b_hat = result_hat = nullptr;
    a_hat_pad = b_hat_pad = a_pad = b_pad = prod_pad = prod_hat_pad = nullptr;
    fwd_orig_a = fwd_orig_b = inv_orig = inv_pad_a = inv_pad_b = fwd_pad = nullptr;

    auto allocate = [](auto& ptr, size_t count, const char* name) {
        ptr = reinterpret_cast<std::complex<double>*>(fftw_malloc(count * sizeof(fftw_complex)));
        if (!ptr) throw std::runtime_error(std::string("fftw_malloc failed for ") + name);
    };

    try {
        allocate(work_orig,   Nx * Ny, "work_orig");
        allocate(a_hat,       Nx * Ny, "a_hat");
        allocate(b_hat,       Nx * Ny, "b_hat");
        allocate(result_hat,  Nx * Ny, "result_hat");
        allocate(a_hat_pad,   Kx * Ky, "a_hat_pad");
        allocate(b_hat_pad,   Kx * Ky, "b_hat_pad");
        allocate(a_pad,       Kx * Ky, "a_pad");
        allocate(b_pad,       Kx * Ky, "b_pad");
        allocate(prod_pad,    Kx * Ky, "prod_pad");
        allocate(prod_hat_pad, Kx * Ky, "prod_hat_pad");
    } catch (const std::runtime_error&) {
        fftw_free(work_orig); fftw_free(a_hat); fftw_free(b_hat); fftw_free(result_hat);
        fftw_free(a_hat_pad); fftw_free(b_hat_pad); fftw_free(a_pad); fftw_free(b_pad);
        fftw_free(prod_pad); fftw_free(prod_hat_pad);
        throw;
    }

    // Create FFTW plans (2D)
    try {
        int n[2] = { static_cast<int>(Nx), static_cast<int>(Ny) };
        fwd_orig_a = fftw_plan_dft_2d(n[0], n[1],
            reinterpret_cast<fftw_complex*>(work_orig),
            reinterpret_cast<fftw_complex*>(a_hat),
            FFTW_FORWARD, planner_effort);
        if (!fwd_orig_a) throw std::runtime_error("fwd_orig_a plan failed");

        fwd_orig_b = fftw_plan_dft_2d(n[0], n[1],
            reinterpret_cast<fftw_complex*>(work_orig),
            reinterpret_cast<fftw_complex*>(b_hat),
            FFTW_FORWARD, planner_effort);
        if (!fwd_orig_b) throw std::runtime_error("fwd_orig_b plan failed");

        inv_orig = fftw_plan_dft_2d(n[0], n[1],
            reinterpret_cast<fftw_complex*>(result_hat),
            reinterpret_cast<fftw_complex*>(work_orig),
            FFTW_BACKWARD, planner_effort);
        if (!inv_orig) throw std::runtime_error("inv_orig plan failed");

        int m[2] = { static_cast<int>(Kx), static_cast<int>(Ky) };
        inv_pad_a = fftw_plan_dft_2d(m[0], m[1],
            reinterpret_cast<fftw_complex*>(a_hat_pad),
            reinterpret_cast<fftw_complex*>(a_pad),
            FFTW_BACKWARD, planner_effort);
        if (!inv_pad_a) throw std::runtime_error("inv_pad_a plan failed");

        inv_pad_b = fftw_plan_dft_2d(m[0], m[1],
            reinterpret_cast<fftw_complex*>(b_hat_pad),
            reinterpret_cast<fftw_complex*>(b_pad),
            FFTW_BACKWARD, planner_effort);
        if (!inv_pad_b) throw std::runtime_error("inv_pad_b plan failed");

        fwd_pad = fftw_plan_dft_2d(m[0], m[1],
            reinterpret_cast<fftw_complex*>(prod_pad),
            reinterpret_cast<fftw_complex*>(prod_hat_pad),
            FFTW_FORWARD, planner_effort);
        if (!fwd_pad) throw std::runtime_error("fwd_pad plan failed");

    } catch (const std::runtime_error&) {
        if (fwd_orig_a) fftw_destroy_plan(fwd_orig_a);
        if (fwd_orig_b) fftw_destroy_plan(fwd_orig_b);
        if (inv_orig)   fftw_destroy_plan(inv_orig);
        if (inv_pad_a)  fftw_destroy_plan(inv_pad_a);
        if (inv_pad_b)  fftw_destroy_plan(inv_pad_b);
        if (fwd_pad)    fftw_destroy_plan(fwd_pad);
        fftw_free(work_orig); fftw_free(a_hat); fftw_free(b_hat); fftw_free(result_hat);
        fftw_free(a_hat_pad); fftw_free(b_hat_pad); fftw_free(a_pad); fftw_free(b_pad);
        fftw_free(prod_pad); fftw_free(prod_hat_pad);
        work_orig = a_hat = b_hat = result_hat = nullptr;
        a_hat_pad = b_hat_pad = a_pad = b_pad = prod_pad = prod_hat_pad = nullptr;
        fwd_orig_a = fwd_orig_b = inv_orig = inv_pad_a = inv_pad_b = fwd_pad = nullptr;
        throw;
    }
}

Dealiaser2D::~Dealiaser2D() {
    fftw_destroy_plan(fwd_orig_a);
    fftw_destroy_plan(fwd_orig_b);
    fftw_destroy_plan(inv_orig);
    fftw_destroy_plan(inv_pad_a);
    fftw_destroy_plan(inv_pad_b);
    fftw_destroy_plan(fwd_pad);
    fftw_free(work_orig);
    fftw_free(a_hat);
    fftw_free(b_hat);
    fftw_free(result_hat);
    fftw_free(a_hat_pad);
    fftw_free(b_hat_pad);
    fftw_free(a_pad);
    fftw_free(b_pad);
    fftw_free(prod_pad);
    fftw_free(prod_hat_pad);
}

void Dealiaser2D::pad_and_copy(const std::complex<double>* src, std::complex<double>* dst) {
    // Zero entire padded array
    std::fill(dst, dst + Kx * Ky, std::complex<double>(0.0, 0.0));
    // Copy four quadrants of low frequencies
    // Quadrant I (top‑left)
    for (size_t i = 0; i < hx; ++i) {
        std::copy(src + i * Ny,
                  src + i * Ny + hy,
                  dst + i * Ky);
    }
    // Quadrant II (top‑right)
    for (size_t i = 0; i < hx; ++i) {
        std::copy(src + i * Ny + Ny - hy,
                  src + i * Ny + Ny,
                  dst + i * Ky + Ky - hy);
    }
    // Quadrant III (bottom‑left)
    for (size_t i = 0; i < hx; ++i) {
        std::copy(src + (Nx - hx + i) * Ny,
                  src + (Nx - hx + i) * Ny + hy,
                  dst + (Kx - hx + i) * Ky);
    }
    // Quadrant IV (bottom‑right)
    for (size_t i = 0; i < hx; ++i) {
        std::copy(src + (Nx - hx + i) * Ny + Ny - hy,
                  src + (Nx - hx + i) * Ny + Ny,
                  dst + (Kx - hx + i) * Ky + Ky - hy);
    }
}

void Dealiaser2D::truncate_and_copy(const std::complex<double>* src, std::complex<double>* dst) {
    // Zero entire original array
    std::fill(dst, dst + Nx * Ny, std::complex<double>(0.0, 0.0));
    // Copy four quadrants from padded spectrum
    for (size_t i = 0; i < hx; ++i) {
        std::copy(src + i * Ky,
                  src + i * Ky + hy,
                  dst + i * Ny);
    }
    for (size_t i = 0; i < hx; ++i) {
        std::copy(src + i * Ky + Ky - hy,
                  src + i * Ky + Ky,
                  dst + i * Ny + Ny - hy);
    }
    for (size_t i = 0; i < hx; ++i) {
        std::copy(src + (Kx - hx + i) * Ky,
                  src + (Kx - hx + i) * Ky + hy,
                  dst + (Nx - hx + i) * Ny);
    }
    for (size_t i = 0; i < hx; ++i) {
        std::copy(src + (Kx - hx + i) * Ky + Ky - hy,
                  src + (Kx - hx + i) * Ky + Ky,
                  dst + (Nx - hx + i) * Ny + Ny - hy);
    }
}

void Dealiaser2D::operator()(const std::complex<double>* a,//Nx*Ny
                             const std::complex<double>* b,//Nx*Ny
                             bool input_physical,
                             bool output_physical,
                             std::complex<double>* result)//Nx*Ny
{
    // Step 1: Fourier representations of a, b
    if (input_physical) {
        std::copy(a, a + Nx * Ny, work_orig);
        fftw_execute(fwd_orig_a);   // work_orig -> a_hat
        std::copy(b, b + Nx * Ny, work_orig);
        fftw_execute(fwd_orig_b);   // work_orig -> b_hat
    } else {
        std::copy(a, a + Nx * Ny, a_hat);
        std::copy(b, b + Nx * Ny, b_hat);
    }

    // Step 2: Pad spectra
    pad_and_copy(a_hat, a_hat_pad);
    pad_and_copy(b_hat, b_hat_pad);

    // Step 3: Inverse FFT to physical space on padded grid (normalized)
    fftw_execute(inv_pad_a);   // a_hat_pad -> a_pad (unnormalized)
    fftw_execute(inv_pad_b);   // b_hat_pad -> b_pad (unnormalized)
    double inv_norm_pad = 1.0 / double(Nx * Ny);
    for (size_t i = 0; i < Kx * Ky; ++i) {
        a_pad[i] *= inv_norm_pad;
        b_pad[i] *= inv_norm_pad;
    }

    // Step 4: Pointwise product in physical space
    for (size_t i = 0; i < Kx * Ky; ++i) {
        prod_pad[i] = a_pad[i] * b_pad[i];
    }

    // Step 5: Forward FFT back to Fourier space (unnormalized)
    fftw_execute(fwd_pad);      // prod_pad -> prod_hat_pad

    // Step 6: Truncate and scale
    truncate_and_copy(prod_hat_pad, result_hat);
    for (size_t i = 0; i < Nx * Ny; ++i) {
        result_hat[i] /= scale;
    }

    // Step 7: Output
    if (!output_physical) {
        std::copy(result_hat, result_hat + Nx * Ny, result);
    } else {
        fftw_execute(inv_orig);   // result_hat -> work_orig (unnormalized)
        double inv_norm_orig = 1.0 / double(Nx * Ny);
        for (size_t i = 0; i < Nx * Ny; ++i) {
            result[i] = work_orig[i] * inv_norm_orig;
        }
    }
}
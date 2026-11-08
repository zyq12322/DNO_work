#include "DNO_calc_OE.h"

#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <cassert>

// external initialization required
std::vector<double> factorial_list;// max(DNO_M, DNO_Mb)>0 ? [0!,1!,...,max(DNO_M, DNO_Mb)!] : [0!,1!]
// \eta  \beta and k pow list in Fourier space.-the index corresponds to the power.
// external initialization required
std::vector<std::vector<std::complex<double>>> eta_pow_hat;//DNO_M > 0 ? (DNO_M + 1)*len(x) : 2*len(x)
std::vector<std::vector<std::complex<double>>> beta_pow_hat;//DNO_Mb > 0 ? (DNO_Mb + 1)*len(x) : 2*len(x)
std::vector<std::vector<std::complex<double>>> D_pow_hat;//max(DNO_M, DNO_Mb)>0 ? (max(DNO_M, DNO_Mb) + 1)*len(x) : 2*len(x)


// G(\beta, \eta)[\xi, \gamma] = \sum G_1^{(m0,m1)}\xi + \sum G_2^(m0,m1)\gamma
// calculate G_1^{(m0,0)}\xi m0 = 0,1,2,3,...,DNO_Mb ; Then fill the results into G1_m0_m1_list
// make sure that G2(m0, 0) \xi has been filled rightly in G2_m0_m1_list[m0*(DNO_M + 1)], m0 = 0, 1, ..., DNO_Mb before call this function.
void G1_m0_0_2d(
    const std::vector<std::complex<double>>& xi_hat,// \xi in Fourier Space
    std::vector<std::vector<std::complex<double>>>& G1_m0_m1_list,//对上标阶数展平的数组[G1(0,0),G1(0,1),G1(1,0),G1(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    const std::vector<std::vector<std::complex<double>>>& G2_m0_m1_list,//对上标阶数展平的数组[G2(0,0),G2(0,1),G2(1,0),G2(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    double h0,// mean depth
    int DNO_M,// max power of \eta;
    int DNO_Mb,// max power of \beta;
    bool dealiase_flag)// flag for dealiasing product-true: use dealiasing for product between two vectors;
{
    if (!fft_obj || !dealiaser) throw std::runtime_error("Global FFT/Dealiaser not initialized");
    size_t N = xi_hat.size();

    for (size_t i = 0; i < N; ++i) {
        // G1(0,0)=D * tanh(h0 * D) \xi
        G1_m0_m1_list[0][i] = D_pow_hat[1][i] * std::tanh(h0 * D_pow_hat[1][i]) * xi_hat[i];
    }


    // loop for the calculation of G1(m0_i,0) with m0_i = 1 ... DNO_Mb
    for (int m0_i = 1; m0_i <= DNO_Mb; ++m0_i) {
        auto& G1_m0_i_0 = G1_m0_m1_list[m0_i * (DNO_M + 1)];
        // initialize the sum result vector G1_m0_i_0
        std::fill(G1_m0_i_0.begin(), G1_m0_i_0.end(), std::complex<double>(0.0, 0.0));

        // 1<= l = odd <= m0_i
        for (int l = 1; l <= m0_i; l += 2) {
            // calculate sech(h0 D)D^l \frac{\beta^l}{l!} D G2(m0-l,0)
            // use G2(m0_i-l,0)
            const auto& row = G2_m0_m1_list[(m0_i - l) * (DNO_M + 1)];
            std::vector<std::complex<double>> odd_term(row.begin(), row.end());
            

            // D
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] *= D_pow_hat[1][i];
            }

            // 1/l!
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] /= factorial_list[l];
            }

            // \beta^l
            if (dealiase_flag) {
                (*dealiaser)(odd_term.data(), beta_pow_hat[l].data(),
                          false,   // input in Fourier space
                          false,   // output in Fourier space
                          odd_term.data());
            }
            else {
                odd_term = fft_obj->ifft(odd_term);
                std::vector<std::complex<double>> beta_phy = fft_obj->ifft(beta_pow_hat[l]);
                for (size_t i = 0; i < N; ++i) {
                    odd_term[i] *= beta_phy[i];
                }
                odd_term = fft_obj->fft(odd_term);
            }

            // sech(h0 D)
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] /= std::cosh(h0 * D_pow_hat[1][i]);
            }

            // D^l
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] *= D_pow_hat[l][i];
            }

            // G1(m0_i,0) -= odd_term
            for (size_t i = 0; i < N; ++i) {
                G1_m0_i_0[i] -= odd_term[i];
            }
        }
    }
}

// G(\beta, \eta)[\xi, \gamma] = \sum G_1^{(m0,m1)}\xi + \sum G_2^(m0,m1)\gamma
// calculate G_2^{(m0,0)}\gamma m0 = 0,1,2,3,...,DNO_Mb ; Then fill the results into G2_m0_m1_list
void G2_m0_0_2d(
    const std::vector<std::complex<double>>& gamma_hat,// \gamma in Fourier Space
    std::vector<std::vector<std::complex<double>>>& G2_m0_m1_list,//对上标阶数展平的数组[G2(0,0),G2(0,1),G2(1,0),G2(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    double h0,// mean depth
    int DNO_M,// max power of \eta;
    int DNO_Mb,// max power of \beta;
    bool dealiase_flag)// flag for dealiasing product-true: use dealiasing for product between two vectors;
{
    if (!fft_obj || !dealiaser) throw std::runtime_error("Global FFT/Dealiaser not initialized");
    size_t N = gamma_hat.size();

    // G2(0,0)=sech(h0 * D) \gamma
    for (size_t i = 0; i < N; ++i) {
        G2_m0_m1_list[0][i] = gamma_hat[i] / std::cosh(h0 * D_pow_hat[1][i]);
    }

    // loop for the calculation of G2(m0_i,0) with m0_i = 1 ... DNO_Mb
    for (int m0_i = 1; m0_i <= DNO_Mb; ++m0_i) {
        auto& G2_m0_i_0 = G2_m0_m1_list[m0_i * (DNO_M + 1)];
        // initialize the sum result vector G2_m0_i_0
        std::fill(G2_m0_i_0.begin(), G2_m0_i_0.end(), std::complex<double>(0.0, 0.0));

        // 1<= l = odd <= m0_i
        for (int l = 1; l <= m0_i; l += 2) {
            // calculate D^{l-1} tanh(h0 D) \frac{\beta^l}{l!} D G2(m0_i-l,0)
            // use G2(m0_i-l,0)
            const auto& row = G2_m0_m1_list[(m0_i - l) * (DNO_M + 1)];
            std::vector<std::complex<double>> odd_term(row.begin(), row.end());

            // D
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] *= D_pow_hat[1][i];
            }

            // 1/l!
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] /= factorial_list[l];
            }

            // \beta^l
            if (dealiase_flag) {
                (*dealiaser)(odd_term.data(), beta_pow_hat[l].data(),
                          false,   // input in Fourier space
                          false,   // output in Fourier space
                          odd_term.data());
            }
            else {
                odd_term = fft_obj->ifft(odd_term);
                std::vector<std::complex<double>> beta_phy = fft_obj->ifft(beta_pow_hat[l]);
                for (size_t i = 0; i < N; ++i) {
                    odd_term[i] *= beta_phy[i];
                }
                odd_term = fft_obj->fft(odd_term);
            }
            
            // tanh(h0 D)
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] *= std::tanh(h0 * D_pow_hat[1][i]);
            }

            // D^{l-1}
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] *= D_pow_hat[l-1][i];
            }

            // G2(m0_i,0) += odd_term
            for (size_t i = 0; i < N; ++i) {
                G2_m0_i_0[i] += odd_term[i];
            }
        }


        // 2<= l = even <= m0_i
        for (int l = 2; l <= m0_i; l += 2) {
            // calculate - D^{l-1} \frac{\beta^l}{l!} D G2(m0_i-l,0)
            // use G2(m0_i-l,0)
            const auto& row = G2_m0_m1_list[(m0_i - l) * (DNO_M + 1)];
            std::vector<std::complex<double>> even_term(row.begin(), row.end());

            // D
            for (size_t i = 0; i < N; ++i) {
                even_term[i] *= D_pow_hat[1][i];
            }

            // 1/l!
            for (size_t i = 0; i < N; ++i) {
                even_term[i] /= factorial_list[l];
            }

            // \beta^l
            if (dealiase_flag) {
                (*dealiaser)(even_term.data(), beta_pow_hat[l].data(),
                          false,   // input in Fourier space
                          false,   // output in Fourier space
                          even_term.data());
            }
            else {
                even_term = fft_obj->ifft(even_term);
                std::vector<std::complex<double>> beta_phy = fft_obj->ifft(beta_pow_hat[l]);
                for (size_t i = 0; i < N; ++i) {
                    even_term[i] *= beta_phy[i];
                }
                even_term = fft_obj->fft(even_term);
            }

            // D^{l-1}
            for (size_t i = 0; i < N; ++i) {
                even_term[i] *= D_pow_hat[l-1][i];
            }

            // G2(m0_i,0) -= even_term
            for (size_t i = 0; i < N; ++i) {
                G2_m0_i_0[i] -= even_term[i];
            }
        }
    }
}

// G(\beta, \eta)[\xi, \gamma] = \sum G_1^{(m0,m1)}\xi + \sum G_2^(m0,m1)\gamma
// calculate G_1^{(0,m1)}\xi m1 = 0,1,2,3,...,DNO_M ; Then fill the results into G1_m0_m1_list
void G1_0_m1_2d(
    const std::vector<std::complex<double>>& xi_hat,// \xi in Fourier Space
    std::vector<std::vector<std::complex<double>>>& G1_m0_m1_list,//对上标阶数展平的数组[G1(0,0),G1(0,1),G1(1,0),G1(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    double h0,// mean depth
    int DNO_M,// max power of \eta;
    int DNO_Mb,// max power of \beta;
    bool dealiase_flag)// flag for dealiasing product-true: use dealiasing for product between two vectors;
{
    if (!fft_obj || !dealiaser) throw std::runtime_error("Global FFT/Dealiaser not initialized");
    size_t N = xi_hat.size();

    for (size_t i = 0; i < N; ++i) {
        // G1(0,0)=D * tanh(h0 * D) \xi
        G1_m0_m1_list[0][i] = D_pow_hat[1][i] * std::tanh(h0 * D_pow_hat[1][i]) * xi_hat[i];
    }

    // loop for the calculation of G1(0,m1_i) with m1_i = 1 ... DNO_M
    for (int m1_i = 1; m1_i <= DNO_M; ++m1_i) {
        /*
            m1_i = even > 0
            G1(0,m1_i) = D^{m1_i} tanh(h0 D) \frac{\eta^{m1_i}}{m1_i!} D
                         -\sum{2<= l=even <= m1_i} D^l \frac{\eta^l}{l!} G1(0,m1_i-l)
                         -\sum{1<= l=odd <= m1_i - 1} D^l tanh(h0 D) \frac{\eta^l}{l!} G1(0,m1_i-l)

            m1_i = odd > 0
            G1(0,m1_i) = D^{m1_i} \frac{\eta^{m1_i}}{m1_i!} D
                         -\sum{2<= l=even <= m1_i - 1} D^l \frac{\eta^l}{l!} G1(0,m1_i-l)
                         -\sum{1<= l=odd <= m1_i} D^l tanh(h0 D) \frac{\eta^l}{l!} G1(0,m1_i-l)
        */
        auto& G1_0_m1_i = G1_m0_m1_list[m1_i];
     
        

        // m1_i = even
        if (m1_i % 2 == 0) {
            // D^{m1_i} tanh(h0 D) \frac{\eta^{m1_i}}{m1_i!} D

            // D
            for (size_t i = 0; i < N; ++i) {
                G1_0_m1_i[i] = xi_hat[i] * D_pow_hat[1][i];
            }

            // \eta^{m1_i}
            if (dealiase_flag) {
                (*dealiaser)(G1_0_m1_i.data(), eta_pow_hat[m1_i].data(),
                          false,   // input in Fourier space
                          false,   // output in Fourier space
                          G1_0_m1_i.data());
            }
            else {
                G1_0_m1_i = fft_obj->ifft(G1_0_m1_i);
                std::vector<std::complex<double>> eta_phy = fft_obj->ifft(eta_pow_hat[m1_i]);
                for (size_t i = 0; i < N; ++i) {
                    G1_0_m1_i[i] *= eta_phy[i];
                }
                G1_0_m1_i = fft_obj->fft(G1_0_m1_i);
            }

            // 1/m1_i!
            for (size_t i = 0; i < N; ++i) {
                G1_0_m1_i[i] /= factorial_list[m1_i];
            }

            // tanh(h0 D)
            for (size_t i = 0; i < N; ++i) {
                G1_0_m1_i[i] *= std::tanh(h0 * D_pow_hat[1][i]);
            }

            // D^{m1_i}
            for (size_t i = 0; i < N; ++i) {
                G1_0_m1_i[i] *= D_pow_hat[m1_i][i];
            }

            
            // - \sum{1<= l=odd <= m1_i-1} D^l tanh(h0 D) \frac{\eta^l}{l!} G1(0,m1_i-l)
            for (int l = 1; l < m1_i; l += 2) {
                // use G1(0,m1_i-l)
                const auto& row = G1_m0_m1_list[m1_i - l];
                std::vector<std::complex<double>> odd_term(row.begin(), row.end());

                // \eta^{l}
                if (dealiase_flag) {
                    (*dealiaser)(odd_term.data(), eta_pow_hat[l].data(),
                            false,   // input in Fourier space
                            false,   // output in Fourier space
                            odd_term.data());
                }
                else {
                    odd_term = fft_obj->ifft(odd_term);
                    std::vector<std::complex<double>> eta_phy = fft_obj->ifft(eta_pow_hat[l]);
                    for (size_t i = 0; i < N; ++i) {
                        odd_term[i] *= eta_phy[i];
                    }
                    odd_term = fft_obj->fft(odd_term);
                }

                // 1/l!
                for (size_t i = 0; i < N; ++i) {
                    odd_term[i] /= factorial_list[l];
                }

                // tanh(h0 D)
                for (size_t i = 0; i < N; ++i) {
                    odd_term[i] *= std::tanh(h0 * D_pow_hat[1][i]);
                }

                // D^{l}
                for (size_t i = 0; i < N; ++i) {
                    odd_term[i] *= D_pow_hat[l][i];
                }

                // G1(0,m1_i) -= odd_term
                for (size_t i = 0; i < N; ++i) {
                    G1_0_m1_i[i] -= odd_term[i];
                }
            }


            // - \sum{2<= l=even <= m1_i} D^l \frac{\eta^l}{l!} G1(0,m1_i-l)
            for (int l = 2; l <= m1_i; l += 2) {
                // use G1(0,m1_i-l)
                const auto& row = G1_m0_m1_list[m1_i - l];
                std::vector<std::complex<double>> even_term(row.begin(), row.end());

                // \eta^{l}
                if (dealiase_flag) {
                    (*dealiaser)(even_term.data(), eta_pow_hat[l].data(),
                            false,   // input in Fourier space
                            false,   // output in Fourier space
                            even_term.data());
                }
                else {
                    even_term = fft_obj->ifft(even_term);
                    std::vector<std::complex<double>> eta_phy = fft_obj->ifft(eta_pow_hat[l]);
                    for (size_t i = 0; i < N; ++i) {
                        even_term[i] *= eta_phy[i];
                    }
                    even_term = fft_obj->fft(even_term);
                }

                // 1/l!
                for (size_t i = 0; i < N; ++i) {
                    even_term[i] /= factorial_list[l];
                }

                // D^{l}
                for (size_t i = 0; i < N; ++i) {
                    even_term[i] *= D_pow_hat[l][i];
                }

                // G1(0,m1_i) -= even_term
                for (size_t i = 0; i < N; ++i) {
                    G1_0_m1_i[i] -= even_term[i];
                }
            }
        }
        else {
            // m1_i = odd
            // D^{m1_i} \frac{\eta^{m1_i}}{m1_i!} D

            // D
            for (size_t i = 0; i < N; ++i) {
                G1_0_m1_i[i] = xi_hat[i] * D_pow_hat[1][i];
            }

            // \eta^{m1_i}
            if (dealiase_flag) {
                (*dealiaser)(G1_0_m1_i.data(), eta_pow_hat[m1_i].data(),
                          false,   // input in Fourier space
                          false,   // output in Fourier space
                          G1_0_m1_i.data());
            }
            else {
                G1_0_m1_i = fft_obj->ifft(G1_0_m1_i);
                std::vector<std::complex<double>> eta_phy = fft_obj->ifft(eta_pow_hat[m1_i]);
                for (size_t i = 0; i < N; ++i) {
                    G1_0_m1_i[i] *= eta_phy[i];
                }
                G1_0_m1_i = fft_obj->fft(G1_0_m1_i);
            }

            // 1/m1_i!
            for (size_t i = 0; i < N; ++i) {
                G1_0_m1_i[i] /= factorial_list[m1_i];
            }

            // D^{m1_i}
            for (size_t i = 0; i < N; ++i) {
                G1_0_m1_i[i] *= D_pow_hat[m1_i][i];
            }

            
            // - \sum{1<= l=odd <= m1_i} D^l tanh(h0 D) \frac{\eta^l}{l!} G1(0,m1_i-l)
            for (int l = 1; l <= m1_i; l += 2) {
                // use G1(0,m1_i-l)
                const auto& row = G1_m0_m1_list[m1_i - l];
                std::vector<std::complex<double>> odd_term(row.begin(), row.end());

                // \eta^{l}
                if (dealiase_flag) {
                    (*dealiaser)(odd_term.data(), eta_pow_hat[l].data(),
                            false,   // input in Fourier space
                            false,   // output in Fourier space
                            odd_term.data());
                }
                else {
                    odd_term = fft_obj->ifft(odd_term);
                    std::vector<std::complex<double>> eta_phy = fft_obj->ifft(eta_pow_hat[l]);
                    for (size_t i = 0; i < N; ++i) {
                        odd_term[i] *= eta_phy[i];
                    }
                    odd_term = fft_obj->fft(odd_term);
                }

                // 1/l!
                for (size_t i = 0; i < N; ++i) {
                    odd_term[i] /= factorial_list[l];
                }

                // tanh(h0 D)
                for (size_t i = 0; i < N; ++i) {
                    odd_term[i] *= std::tanh(h0 * D_pow_hat[1][i]);
                }

                // D^{l}
                for (size_t i = 0; i < N; ++i) {
                    odd_term[i] *= D_pow_hat[l][i];
                }

                // G1(0,m1_i) -= odd_term
                for (size_t i = 0; i < N; ++i) {
                    G1_0_m1_i[i] -= odd_term[i];
                }
            }


            // - \sum{2<= l=even <= m1_i-1} D^l \frac{\eta^l}{l!} G1(0,m1_i-l)
            for (int l = 2; l < m1_i; l += 2) {
                // use G1(0,m1_i-l)
                const auto& row = G1_m0_m1_list[m1_i - l];
                std::vector<std::complex<double>> even_term(row.begin(), row.end());

                // \eta^{l}
                if (dealiase_flag) {
                    (*dealiaser)(even_term.data(), eta_pow_hat[l].data(),
                            false,   // input in Fourier space
                            false,   // output in Fourier space
                            even_term.data());
                }
                else {
                    even_term = fft_obj->ifft(even_term);
                    std::vector<std::complex<double>> eta_phy = fft_obj->ifft(eta_pow_hat[l]);
                    for (size_t i = 0; i < N; ++i) {
                        even_term[i] *= eta_phy[i];
                    }
                    even_term = fft_obj->fft(even_term);
                }

                // 1/l!
                for (size_t i = 0; i < N; ++i) {
                    even_term[i] /= factorial_list[l];
                }

                // D^{l}
                for (size_t i = 0; i < N; ++i) {
                    even_term[i] *= D_pow_hat[l][i];
                }

                // G1(0,m1_i) -= even_term
                for (size_t i = 0; i < N; ++i) {
                    G1_0_m1_i[i] -= even_term[i];
                }
            }
        }
    }
}

// G(\beta, \eta)[\xi, \gamma] = \sum G_1^{(m0,m1)}\xi + \sum G_2^(m0,m1)\gamma
// calculate G_2^{(0,m1)}\gamma m1 = 0,1,2,3,...,DNO_M ; Then fill the results into G2_m0_m1_list
// make sure that G1(0, m1) \gamma has been filled rightly in G1_m0_m1_list[m1], m1 = 0, 1, ..., DNO_M before call this function.
void G2_0_m1_2d(
    const std::vector<std::complex<double>>& gamma_hat,// \gamma in Fourier Space
    const std::vector<std::vector<std::complex<double>>>& G1_m0_m1_list,//对上标阶数展平的数组[G1(0,0),G1(0,1),G1(1,0),G1(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    std::vector<std::vector<std::complex<double>>>& G2_m0_m1_list,//对上标阶数展平的数组[G2(0,0),G2(0,1),G2(1,0),G2(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    double h0,// mean depth
    int DNO_M,// max power of \eta;
    int DNO_Mb,// max power of \beta;
    bool dealiase_flag)// flag for dealiasing product-true: use dealiasing for product between two vectors;
{
    if (!fft_obj || !dealiaser) throw std::runtime_error("Global FFT/Dealiaser not initialized");
    size_t N = gamma_hat.size();

    // G2(0,0)=sech(h0 * D) \gamma
    for (size_t i = 0; i < N; ++i) {
        G2_m0_m1_list[0][i] = gamma_hat[i] / std::cosh(h0 * D_pow_hat[1][i]);
    }

    // loop for the calculation of G2(0,m1_i) with m1_i = 1 ... DNO_M
    for (int m1_i = 1; m1_i <= DNO_M; ++m1_i) {
        /*
            m1_i = even > 0
            G2(0,m1_i) = D^{m1_i-1} sech(h0 D) \frac{\eta^{m1_i}}{m1_i!} D
                         -\sum{1<= l=odd <= m1_i - 1} D^{l-1} sech(h0 D) \frac{\eta^l}{l!} G1(0,m1_i-l)

            m1_i = odd > 0
            G2(0,m1_i) = -\sum{1<= l=odd <= m1_i} D^{l-1} sech(h0 D) \frac{\eta^l}{l!} G1(0,m1_i-l)
        */
        auto& G2_0_m1_i = G2_m0_m1_list[m1_i];

        // m1_i = even
        if (m1_i % 2 == 0) {
            // D^{m1_i-1} sech(h0 D) \frac{\eta^{m1_i}}{m1_i!} D

            // D
            for (size_t i = 0; i < N; ++i) {
                G2_0_m1_i[i] = gamma_hat[i] * D_pow_hat[1][i];
            }

            // \eta^{m1_i}
            if (dealiase_flag) {
                (*dealiaser)(G2_0_m1_i.data(), eta_pow_hat[m1_i].data(),
                          false,   // input in Fourier space
                          false,   // output in Fourier space
                          G2_0_m1_i.data());
            }
            else {
                G2_0_m1_i = fft_obj->ifft(G2_0_m1_i);
                std::vector<std::complex<double>> eta_phy = fft_obj->ifft(eta_pow_hat[m1_i]);
                for (size_t i = 0; i < N; ++i) {
                    G2_0_m1_i[i] *= eta_phy[i];
                }
                G2_0_m1_i = fft_obj->fft(G2_0_m1_i);
            }

            // 1/m1_i!
            for (size_t i = 0; i < N; ++i) {
                G2_0_m1_i[i] /= factorial_list[m1_i];
            }

            // sech(h0 D)
            for (size_t i = 0; i < N; ++i) {
                G2_0_m1_i[i] /= std::cosh(h0 * D_pow_hat[1][i]);
            }

            // D^{m1_i-1}
            for (size_t i = 0; i < N; ++i) {
                G2_0_m1_i[i] *= D_pow_hat[m1_i-1][i];
            }

            
            // - \sum{1<= l=odd <= m1_i-1} D^{l-1} sech(h0 D) \frac{\eta^l}{l!} G1(0,m1_i-l)
            for (int l = 1; l < m1_i; l += 2) {
                // use G1(0,m1_i-l)
                const auto& row = G1_m0_m1_list[m1_i - l];
                std::vector<std::complex<double>> odd_term(row.begin(), row.end());

                // \eta^{l}
                if (dealiase_flag) {
                    (*dealiaser)(odd_term.data(), eta_pow_hat[l].data(),
                            false,   // input in Fourier space
                            false,   // output in Fourier space
                            odd_term.data());
                }
                else {
                    odd_term = fft_obj->ifft(odd_term);
                    std::vector<std::complex<double>> eta_phy = fft_obj->ifft(eta_pow_hat[l]);
                    for (size_t i = 0; i < N; ++i) {
                        odd_term[i] *= eta_phy[i];
                    }
                    odd_term = fft_obj->fft(odd_term);
                }

                // 1/l!
                for (size_t i = 0; i < N; ++i) {
                    odd_term[i] /= factorial_list[l];
                }

                // sech(h0 D)
                for (size_t i = 0; i < N; ++i) {
                    odd_term[i] /= std::cosh(h0 * D_pow_hat[1][i]);
                }

                // D^{l-1}
                for (size_t i = 0; i < N; ++i) {
                    odd_term[i] *= D_pow_hat[l-1][i];
                }

                // G2(0,m1_i) -= odd_term
                for (size_t i = 0; i < N; ++i) {
                    G2_0_m1_i[i] -= odd_term[i];
                }
            }
        }
        else {
            // m1_i = odd
            // initialize as zeros for sum
            std::fill(G2_0_m1_i.begin(), G2_0_m1_i.end(), std::complex<double>(0.0, 0.0));

            // - \sum{1<= l=odd <= m1_i} D^{l-1} sech(h0 D) \frac{\eta^l}{l!} G1(0,m1_i-l)
            for (int l = 1; l <= m1_i; l += 2) {
                // use G1(0,m1_i-l)
                const auto& row = G1_m0_m1_list[m1_i - l];
                std::vector<std::complex<double>> odd_term(row.begin(), row.end());

                // \eta^{l}
                if (dealiase_flag) {
                    (*dealiaser)(odd_term.data(), eta_pow_hat[l].data(),
                            false,   // input in Fourier space
                            false,   // output in Fourier space
                            odd_term.data());
                }
                else {
                    odd_term = fft_obj->ifft(odd_term);
                    std::vector<std::complex<double>> eta_phy = fft_obj->ifft(eta_pow_hat[l]);
                    for (size_t i = 0; i < N; ++i) {
                        odd_term[i] *= eta_phy[i];
                    }
                    odd_term = fft_obj->fft(odd_term);
                }

                // 1/l!
                for (size_t i = 0; i < N; ++i) {
                    odd_term[i] /= factorial_list[l];
                }

                // sech(h0 D)
                for (size_t i = 0; i < N; ++i) {
                    odd_term[i] /= std::cosh(h0 * D_pow_hat[1][i]);
                }

                // D^{l-1}
                for (size_t i = 0; i < N; ++i) {
                    odd_term[i] *= D_pow_hat[l-1][i];
                }

                // G2(0,m1_i) -= odd_term
                for (size_t i = 0; i < N; ++i) {
                    G2_0_m1_i[i] -= odd_term[i];
                }
            }
        }
    }
}

// G(\beta, \eta)[\xi, \gamma] = \sum G_1^{(m0,m1)}\xi + \sum G_2^(m0,m1)\gamma
// calculate G_1^{(m0,m1)}\xi ,  m0 = m0; m1 = 1, 2, ..., DNO_M; Then fill the results into G1_m0_m1_list
// make sure that G1(m0, 0)\xi  G2(m0_i, m1)\xi (m0_i=0,1,...,m0-1; m1=0,1,...,DNO_M) has been filled rightly before call this function.
void G1_m0_m1_2d(
    int m0, // power of beta for G1 which will be calculated (must be positive)
    const std::vector<std::complex<double>>& xi_hat,// \xi in Fourier Space
    std::vector<std::vector<std::complex<double>>>& G1_m0_m1_list,//对上标阶数展平的数组[G1(0,0),G1(0,1),G1(1,0),G1(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    const std::vector<std::vector<std::complex<double>>>& G2_m0_m1_list,//对上标阶数展平的数组[G2(0,0),G2(0,1),G2(1,0),G2(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    double h0,// mean depth
    int DNO_M,// max power of \eta;
    int DNO_Mb,// max power of \beta;
    bool dealiase_flag)// flag for dealiasing product-true: use dealiasing for product between two vectors;
{
    if (m0 <= 0) {
        throw std::invalid_argument("m0 must be positive in G1_m0_m1_2d");
    }
    if (!fft_obj || !dealiaser) throw std::runtime_error("Global FFT/Dealiaser not initialized");
    size_t N = xi_hat.size();

    // loop for the calculation of G1(m0,m1_i) with m1_i = 1 ... DNO_M
    for (int m1_i = 1; m1_i <= DNO_M; ++m1_i) {
        /*
            m0>0, m1_i>0
            G1(m0,m1_i) = -\sum{2<= l=even <= m1_i} D^{l} \frac{\eta^l}{l!} G1(m0,m1_i-l)
                         -\sum{1<= l=odd <= m1_i} D^{l} tanh(h0 D) \frac{\eta^l}{l!} G1(m0,m1_i-l)
                         -\sum{1<= l=odd <= m0} D^{l} sech(h0 D) \frac{\beta^l}{l!} D G2(m0-l,m1_i)
        */
        auto& G1_m0_m1_i = G1_m0_m1_list[m0*(DNO_M + 1) + m1_i];
        // initialize as zeros for sum
        std::fill(G1_m0_m1_i.begin(), G1_m0_m1_i.end(), std::complex<double>(0.0, 0.0));

        // - \sum{2<= l=even <= m1_i} D^{l} \frac{\eta^l}{l!} G1(m0,m1_i-l)
        for (int l = 2; l <= m1_i; l += 2) {
            // use G1(m0,m1_i-l)
            const auto& row = G1_m0_m1_list[m0*(DNO_M + 1) + m1_i - l];
            std::vector<std::complex<double>> even_term(row.begin(), row.end());

            // \eta^{l}
            if (dealiase_flag) {
                (*dealiaser)(even_term.data(), eta_pow_hat[l].data(),
                        false,   // input in Fourier space
                        false,   // output in Fourier space
                        even_term.data());
            }
            else {
                even_term = fft_obj->ifft(even_term);
                std::vector<std::complex<double>> eta_phy = fft_obj->ifft(eta_pow_hat[l]);
                for (size_t i = 0; i < N; ++i) {
                    even_term[i] *= eta_phy[i];
                }
                even_term = fft_obj->fft(even_term);
            }

            // 1/l!
            for (size_t i = 0; i < N; ++i) {
                even_term[i] /= factorial_list[l];
            }

            // D^{l}
            for (size_t i = 0; i < N; ++i) {
                even_term[i] *= D_pow_hat[l][i];
            }

            // G1(m0,m1_i) -= even_term
            for (size_t i = 0; i < N; ++i) {
                G1_m0_m1_i[i] -= even_term[i];
            }
            
        }

        // - \sum{1<= l=odd <= m1_i} D^{l} tanh(h0 D) \frac{\eta^l}{l!} G1(m0,m1_i-l)
        for (int l = 1; l <= m1_i; l += 2) {
            // use G1(m0,m1_i-l)
            const auto& row = G1_m0_m1_list[m0*(DNO_M + 1) + m1_i - l];
            std::vector<std::complex<double>> odd_term(row.begin(), row.end());

            // \eta^{l}
            if (dealiase_flag) {
                (*dealiaser)(odd_term.data(), eta_pow_hat[l].data(),
                        false,   // input in Fourier space
                        false,   // output in Fourier space
                        odd_term.data());
            }
            else {
                odd_term = fft_obj->ifft(odd_term);
                std::vector<std::complex<double>> eta_phy = fft_obj->ifft(eta_pow_hat[l]);
                for (size_t i = 0; i < N; ++i) {
                    odd_term[i] *= eta_phy[i];
                }
                odd_term = fft_obj->fft(odd_term);
            }

            // 1/l!
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] /= factorial_list[l];
            }

            // tanh(h0 D)
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] *= std::tanh(h0 * D_pow_hat[1][i]);
            }

            // D^{l}
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] *= D_pow_hat[l][i];
            }

            // G1(m0,m1_i) -= odd_term
            for (size_t i = 0; i < N; ++i) {
                G1_m0_m1_i[i] -= odd_term[i];
            }
        }

        // - \sum{1<= l=odd <= m0} D^{l} sech(h0 D) \frac{\beta^l}{l!} D G2(m0-l,m1_i)
        for (int l = 1; l <= m0; l += 2) {
            // use G2(m0-l,m1_i)
            const auto& row = G2_m0_m1_list[(m0 - l)*(DNO_M + 1) + m1_i];
            std::vector<std::complex<double>> odd_term(row.begin(), row.end());

            // D
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] *= D_pow_hat[1][i];
            }

            // 1/l!
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] /= factorial_list[l];
            }

            // \beta^{l}
            if (dealiase_flag) {
                (*dealiaser)(odd_term.data(), beta_pow_hat[l].data(),
                        false,   // input in Fourier space
                        false,   // output in Fourier space
                        odd_term.data());
            }
            else {
                odd_term = fft_obj->ifft(odd_term);
                std::vector<std::complex<double>> beta_phy = fft_obj->ifft(beta_pow_hat[l]);
                for (size_t i = 0; i < N; ++i) {
                    odd_term[i] *= beta_phy[i];
                }
                odd_term = fft_obj->fft(odd_term);
            }

            // sech(h0 D)
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] /= std::cosh(h0 * D_pow_hat[1][i]);
            }

            // D^{l}
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] *= D_pow_hat[l][i];
            }

            // G1(m0,m1_i) -= odd_term
            for (size_t i = 0; i < N; ++i) {
                G1_m0_m1_i[i] -= odd_term[i];
            }
        }
        
    }
}

// G(\beta, \eta)[\xi, \gamma] = \sum G_1^{(m0,m1)}\xi + \sum G_2^(m0,m1)\gamma
// calculate G_2^{(m0,m1)}\gamma ,  m0 = m0; m1 = 1, 2, ..., DNO_M; Then fill the results into G2_m0_m1_list
// make sure that G1(m0, m1_i)\gamma (m1_i=0,1,...,DNO_M)  G2(m0_i, m1_i)\gamma (m0_i=0,1,...,m0-1; m1_i=0,1,...,DNO_M) has been filled rightly before call this function.
void G2_m0_m1_2d(
    int m0, // power of beta for G2 which will be calculated (must be positive)
    const std::vector<std::complex<double>>& gamma_hat,// \gamma in Fourier Space
    const std::vector<std::vector<std::complex<double>>>& G1_m0_m1_list,//对上标阶数展平的数组[G1(0,0),G1(0,1),G1(1,0),G1(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    std::vector<std::vector<std::complex<double>>>& G2_m0_m1_list,//对上标阶数展平的数组[G2(0,0),G2(0,1),G2(1,0),G2(1,1)] shape:((DNO_Mb + 1)*(DNO_M + 1), len(x))
    double h0,// mean depth
    int DNO_M,// max power of \eta;
    int DNO_Mb,// max power of \beta;
    bool dealiase_flag)// flag for dealiasing product-true: use dealiasing for product between two vectors;
{
    if (m0 <= 0) {
        throw std::invalid_argument("m0 must be positive in G2_m0_m1_2d");
    }
    if (!fft_obj || !dealiaser) throw std::runtime_error("Global FFT/Dealiaser not initialized");
    size_t N = gamma_hat.size();

    // loop for the calculation of G2(m0,m1_i) with m1_i = 1 ... DNO_M
    for (int m1_i = 1; m1_i <= DNO_M; ++m1_i) {
        /*
            m0>0, m1_i>0
            G2(m0,m1_i) = -\sum{2<= l=even <= m0} D^{l-1} \frac{\beta^l}{l!} D G2(m0-l,m1_i)
                         -\sum{1<= l=odd <= m1_i} D^{l-1} sech(h0 D) \frac{\eta^l}{l!} G1(m0,m1_i-l)
                         +\sum{1<= l=odd <= m0} D^{l-1} tanh(h0 D) \frac{\beta^l}{l!} D G2(m0-l,m1_i)
        */
        auto& G2_m0_m1_i = G2_m0_m1_list[m0*(DNO_M + 1) + m1_i];
        // initialize as zeros for sum
        std::fill(G2_m0_m1_i.begin(), G2_m0_m1_i.end(), std::complex<double>(0.0, 0.0));

        // - \sum{2<= l=even <= m0} D^{l-1} \frac{\beta^l}{l!} D G2(m0-l,m1_i)
        for (int l = 2; l <= m0; l += 2) {
            // use G2(m0-l,m1_i)
            const auto& row = G2_m0_m1_list[(m0 - l)*(DNO_M + 1) + m1_i];
            std::vector<std::complex<double>> even_term(row.begin(), row.end());

            // D
            for (size_t i = 0; i < N; ++i) {
                even_term[i] *= D_pow_hat[1][i];
            }

            // 1/l!
            for (size_t i = 0; i < N; ++i) {
                even_term[i] /= factorial_list[l];
            }

            // \beta^{l}
            if (dealiase_flag) {
                (*dealiaser)(even_term.data(), beta_pow_hat[l].data(),
                        false,   // input in Fourier space
                        false,   // output in Fourier space
                        even_term.data());
            }
            else {
                even_term = fft_obj->ifft(even_term);
                std::vector<std::complex<double>> beta_phy = fft_obj->ifft(beta_pow_hat[l]);
                for (size_t i = 0; i < N; ++i) {
                    even_term[i] *= beta_phy[i];
                }
                even_term = fft_obj->fft(even_term);
            }

            // D^{l-1}
            for (size_t i = 0; i < N; ++i) {
                even_term[i] *= D_pow_hat[l-1][i];
            }

            // G2(m0,m1_i) -= even_term
            for (size_t i = 0; i < N; ++i) {
                G2_m0_m1_i[i] -= even_term[i];
            }
            
        }


        // - \sum{1<= l=odd <= m1_i} D^{l-1} sech(h0 D) \frac{\eta^l}{l!} G1(m0,m1_i-l)
        for (int l = 1; l <= m1_i; l += 2) {
            // use G1(m0,m1_i-l)
            const auto& row = G1_m0_m1_list[m0*(DNO_M + 1) + m1_i - l];
            std::vector<std::complex<double>> odd_term(row.begin(), row.end());

            // \eta^{l}
            if (dealiase_flag) {
                (*dealiaser)(odd_term.data(), eta_pow_hat[l].data(),
                        false,   // input in Fourier space
                        false,   // output in Fourier space
                        odd_term.data());
            }
            else {
                odd_term = fft_obj->ifft(odd_term);
                std::vector<std::complex<double>> eta_phy = fft_obj->ifft(eta_pow_hat[l]);
                for (size_t i = 0; i < N; ++i) {
                    odd_term[i] *= eta_phy[i];
                }
                odd_term = fft_obj->fft(odd_term);
            }

            // 1/l!
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] /= factorial_list[l];
            }

            // sech(h0 D)
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] /= std::cosh(h0 * D_pow_hat[1][i]);
            }

            // D^{l-1}
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] *= D_pow_hat[l-1][i];
            }

            // G2(m0,m1_i) -= odd_term
            for (size_t i = 0; i < N; ++i) {
                G2_m0_m1_i[i] -= odd_term[i];
            }
        }

        // + \sum{1<= l=odd <= m0} D^{l-1} tanh(h0 D) \frac{\beta^l}{l!} D G2(m0-l,m1_i)
        for (int l = 1; l <= m0; l += 2) {
            // use G2(m0-l,m1_i)
            const auto& row = G2_m0_m1_list[(m0 - l)*(DNO_M + 1) + m1_i];
            std::vector<std::complex<double>> odd_term(row.begin(), row.end());

            // D
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] *= D_pow_hat[1][i];
            }

            // 1/l!
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] /= factorial_list[l];
            }

            // \beta^{l}
            if (dealiase_flag) {
                (*dealiaser)(odd_term.data(), beta_pow_hat[l].data(),
                        false,   // input in Fourier space
                        false,   // output in Fourier space
                        odd_term.data());
            }
            else {
                odd_term = fft_obj->ifft(odd_term);
                std::vector<std::complex<double>> beta_phy = fft_obj->ifft(beta_pow_hat[l]);
                for (size_t i = 0; i < N; ++i) {
                    odd_term[i] *= beta_phy[i];
                }
                odd_term = fft_obj->fft(odd_term);
            }

            // tanh(h0 D)
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] *= std::tanh(h0 * D_pow_hat[1][i]);
            }

            // D^{l-1}
            for (size_t i = 0; i < N; ++i) {
                odd_term[i] *= D_pow_hat[l-1][i];
            }

            // G2(m0,m1_i) += odd_term
            for (size_t i = 0; i < N; ++i) {
                G2_m0_m1_i[i] += odd_term[i];
            }
        }
    }
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
    bool dealiase_flag)// flag for dealiasing product-true: use dealiasing for product between two vectors;
{
    if (xi_hat.size() != G1_xi_hat.size()) throw std::invalid_argument("size of G1_xi_hat is wrong in G1_2d");
    // fill G_1^{(0,m1)}\xi m1 = 0,1,2,3,...,DNO_M ;
    G1_0_m1_2d(xi_hat, G1_m0_m1_list, h0, DNO_M, DNO_Mb, dealiase_flag);
    // fill G_2^{(m0,0)}\xi m0 = 0,1,2,3,...,DNO_Mb ;
    G2_m0_0_2d(xi_hat, G2_m0_m1_list, h0, DNO_M, DNO_Mb, dealiase_flag);
    // fill G_2^{(0,m1)}\xi m1 = 0,1,2,3,...,DNO_M ;
    G2_0_m1_2d(xi_hat, G1_m0_m1_list, G2_m0_m1_list, h0, DNO_M, DNO_Mb, dealiase_flag);
    // fill G_1^{(m0,0)}\xi m0 = 0,1,2,3,...,DNO_Mb ;
    G1_m0_0_2d(xi_hat, G1_m0_m1_list, G2_m0_m1_list, h0, DNO_M, DNO_Mb, dealiase_flag);
    // fill G_1^{(m0_i,m1)}\xi m0_i = 1,2,3,...,DNO_Mb ; m1 = 1,2,3,...,DNO_M ;
    // fill G_2^{(m0_i,m1)}\xi m0_i = 1,2,3,...,DNO_Mb ; m1 = 1,2,3,...,DNO_M ;
    for (int m0_i = 1;m0_i <= DNO_Mb; ++m0_i ) {
        G1_m0_m1_2d(m0_i, xi_hat, G1_m0_m1_list, G2_m0_m1_list, h0, DNO_M, DNO_Mb, dealiase_flag);
        G2_m0_m1_2d(m0_i, xi_hat, G1_m0_m1_list, G2_m0_m1_list, h0, DNO_M, DNO_Mb, dealiase_flag);
    }
    std::fill(G1_xi_hat.begin(), G1_xi_hat.end(), std::complex<double>(0.0, 0.0));
    for (int mi = 0; mi < (DNO_Mb + 1)* (DNO_M + 1); ++mi) {
        for (size_t ind = 0 ; ind < xi_hat.size(); ++ind) {
            G1_xi_hat[ind] += G1_m0_m1_list[mi][ind];
        }
    }
}

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
    bool dealiase_flag)// flag for dealiasing product-true: use dealiasing for product between two vectors;
{
    if (gamma_hat.size() != G2_gamma_hat.size()) throw std::invalid_argument("size of G2_gamma_hat is wrong in G2_2d");
    // fill G_1^{(0,m1)}\gamma m1 = 0,1,2,3,...,DNO_M ;
    G1_0_m1_2d(gamma_hat, G1_m0_m1_list, h0, DNO_M, DNO_Mb, dealiase_flag);
    // fill G_2^{(m0,0)}\gamma m0 = 0,1,2,3,...,DNO_Mb ;
    G2_m0_0_2d(gamma_hat, G2_m0_m1_list, h0, DNO_M, DNO_Mb, dealiase_flag);
    // fill G_2^{(0,m1)}\gamma m1 = 0,1,2,3,...,DNO_M ;
    G2_0_m1_2d(gamma_hat, G1_m0_m1_list, G2_m0_m1_list, h0, DNO_M, DNO_Mb, dealiase_flag);
    // fill G_1^{(m0,0)}\gamma m0 = 0,1,2,3,...,DNO_Mb ;
    G1_m0_0_2d(gamma_hat, G1_m0_m1_list, G2_m0_m1_list, h0, DNO_M, DNO_Mb, dealiase_flag);
    // fill G_1^{(m0_i,m1)}\gamma m0_i = 1,2,3,...,DNO_Mb ; m1 = 1,2,3,...,DNO_M ;
    // fill G_2^{(m0_i,m1)}\gamma m0_i = 1,2,3,...,DNO_Mb ; m1 = 1,2,3,...,DNO_M ;
    for (int m0_i = 1;m0_i <= DNO_Mb; ++m0_i ) {
        G1_m0_m1_2d(m0_i, gamma_hat, G1_m0_m1_list, G2_m0_m1_list, h0, DNO_M, DNO_Mb, dealiase_flag);
        G2_m0_m1_2d(m0_i, gamma_hat, G1_m0_m1_list, G2_m0_m1_list, h0, DNO_M, DNO_Mb, dealiase_flag);
    }
    std::fill(G2_gamma_hat.begin(), G2_gamma_hat.end(), std::complex<double>(0.0, 0.0));
    for (int mi = 0; mi < (DNO_Mb + 1)* (DNO_M + 1); ++mi) {
        for (size_t ind = 0 ; ind < gamma_hat.size(); ++ind) {
            G2_gamma_hat[ind] += G2_m0_m1_list[mi][ind];
        }
    }
}

// \partial_eta_partial_t = G(\beta, \eta)[\xi, \gamma] = \sum G_1^{(m0,m1)}\xi + \sum G_2^(m0,m1)\gamma
// \gamma = \partial_t \beta is known
// make sure eta_pow_hat/beta_pow_hat/D_pow_hat/factorial_list has been refreshed before call this function.
void partial_eta_partial_t_2d(
    const std::vector<std::complex<double>>& xi_hat,
    const std::vector<std::complex<double>>& gamma_hat,
    std::vector<std::complex<double>>& eta_t_hat, // fill this with the result G(\beta, \eta)[\xi, \gamma]
    double h0, // mean_depth
    int DNO_M,
    int DNO_Mb,
    bool dealiase_flag,
    DNOBufferCache* cache)      // ← 新增可选参数
{
    // check size
    const size_t N = xi_hat.size();

    if (DNO_M < 0 || DNO_Mb < 0 || N == 0) {
        throw std::invalid_argument(
            "DNO_M and DNO_Mb must be non-negative and N must be positive in partial_eta_partial_t_2d");
    }

    if (gamma_hat.size() != N) {
        throw std::invalid_argument(
            "size mismatch in partial_eta_partial_t_2d");
    }
    // allocate correct size of memory for vector to store result of \partial_t \eta
    if (eta_t_hat.size() != N) {
        eta_t_hat.resize(N);
    }

    // 用提供的缓存或本地临时对象
    // 未传入 cache 时使用函数级静态缓存：N/DNO_M/DNO_Mb 不变时 ensure_size 为空操作，
    // 避免每次调用都重新分配全部中间缓冲区。
    static thread_local DNOBufferCache local_cache;
    DNOBufferCache& C = cache ? *cache : local_cache;
    C.ensure_size(N, DNO_M, DNO_Mb);

    G1_2d(xi_hat, C.G1_xi_hat,
          C.G1_m0_m1_list, C.G2_m0_m1_list,
          h0, DNO_M, DNO_Mb, dealiase_flag);

    G2_2d(gamma_hat, C.G2_gamma_hat,
          C.G1_m0_m1_list, C.G2_m0_m1_list,
          h0, DNO_M, DNO_Mb, dealiase_flag);

    for (size_t i = 0; i < N; ++i) {
        eta_t_hat[i] = C.G1_xi_hat[i] + C.G2_gamma_hat[i];
    }
}


// \partial_xi_partial_t = -g \eta - 0.5 * (\xi_x^2 - G^2 - 2 \eta_x \xi_x G) / (1 + \eta_x^2)
// make sure eta_pow_hat/beta_pow_hat/D_pow_hat/factorial_list has been refreshed before call this function.
void partial_xi_partial_t_2d(
    const std::vector<std::complex<double>>& xi_hat,
    const std::vector<std::complex<double>>& G_hat,
    std::vector<std::complex<double>>& xi_t_hat,
    double g,
    int DNO_M,
    int DNO_Mb,
    bool dealiase_flag,
    DNOBufferCache* cache)      // ← 新增可选参数
{
    if (!fft_obj || !dealiaser) {
        throw std::runtime_error("Global fft_op/Dealiaser object not initialized");
    }

    // check size
    const size_t N = xi_hat.size();
    if (DNO_M < 0 || DNO_Mb < 0 || N == 0) {
        throw std::invalid_argument(
            "DNO_M and DNO_Mb must be non-negative and N must be positive in partial_xi_partial_t_2d");
    }
    
    if (eta_pow_hat.size() < 2 || eta_pow_hat[1].size() != N) {
        throw std::invalid_argument(
            "eta_pow_hat size error in partial_xi_partial_t_2d");
    }
    if (G_hat.size() != N) {
        throw std::invalid_argument(
            "size mismatch in partial_xi_partial_t_2d");
    }

    // allocate correct size of memory for vector to store result of \partial_t \xi
    if (xi_t_hat.size() != N) {
        xi_t_hat.resize(N);
    }
    

    static thread_local DNOBufferCache local_cache;
    DNOBufferCache& C = cache ? *cache : local_cache;
    C.ensure_size(N, DNO_M, DNO_Mb);

    const auto& eta_hat = eta_pow_hat[1];

    // η 物理空间表示
    C.eta_phy = fft_obj->ifft(eta_hat);

    if (!dealiase_flag) {
        // ---------- 物理空间直接计算 ----------
        C.xi_phy = fft_obj->ifft(xi_hat);
        C.G_phy  = fft_obj->ifft(G_hat);

        for (size_t i = 0; i < N; ++i) {
            C.eta_x_hat[i] = ImUnit * D_pow_hat[1][i] * eta_hat[i];
            C.xi_x_hat[i]  = ImUnit * D_pow_hat[1][i] * xi_hat[i];
        }
        C.eta_x_phy = fft_obj->ifft(C.eta_x_hat);
        C.xi_x_phy  = fft_obj->ifft(C.xi_x_hat);

        for (size_t i = 0; i < N; ++i) {
            const double eta_val   = C.eta_phy[i].real();
            const double eta_x_val = C.eta_x_phy[i].real();
            const double xi_x_val  = C.xi_x_phy[i].real();
            const double G_val     = C.G_phy[i].real();

            const double term1 = xi_x_val * xi_x_val - G_val * G_val
                               - 2.0 * eta_x_val * xi_x_val * G_val;
            const double term2 = 1.0 + eta_x_val * eta_x_val;
            C.rhs_phy[i] = std::complex<double>(
                -g * eta_val - 0.5 * term1 / term2, 0.0);
        }
        xi_t_hat = fft_obj->fft(C.rhs_phy);
    } else {
        // ---------- dealiaser 分支 ----------
        for (size_t i = 0; i < N; ++i) {
            C.eta_x_hat[i] = ImUnit * D_pow_hat[1][i] * eta_hat[i];
            C.xi_x_hat[i]  = ImUnit * D_pow_hat[1][i] * xi_hat[i];
        }

        (*dealiaser)(C.eta_x_hat.data(), C.eta_x_hat.data(), false, false, C.eta_x_2_hat.data());
        (*dealiaser)(C.xi_x_hat.data(),  C.xi_x_hat.data(),  false, false, C.xi_x_2_hat.data());
        (*dealiaser)(G_hat.data(),       G_hat.data(),       false, false, C.G_2_hat.data());
        (*dealiaser)(C.eta_x_hat.data(), C.xi_x_hat.data(),  false, false, C.eta_x_xi_x_hat.data());
        (*dealiaser)(C.eta_x_xi_x_hat.data(), G_hat.data(),  false, false, C.eta_x_xi_x_G_hat.data());

        for (size_t i = 0; i < N; ++i) {
            C.term1_hat[i] = C.xi_x_2_hat[i] - C.G_2_hat[i]
                           - 2.0 * C.eta_x_xi_x_G_hat[i];
        }
        C.term1_phy = fft_obj->ifft(C.term1_hat);

        C.term2_phy = fft_obj->ifft(C.eta_x_2_hat);
        for (size_t i = 0; i < N; ++i) C.term2_phy[i] += 1.0;

        for (size_t i = 0; i < N; ++i) {
            C.rhs_phy[i] = -g * C.eta_phy[i]
                         - 0.5 * C.term1_phy[i] / C.term2_phy[i];
        }
        xi_t_hat = fft_obj->fft(C.rhs_phy);
    }
}
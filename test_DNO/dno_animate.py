#!/usr/bin/env python3
"""
dno_animate.py — 从 DNO 输出文件生成 η–x 与 β–x 的演化视频/动图（单 Y 轴）
                 并可选地在图例中显示每时刻的能量/动量（EVI.txt）

输入：
    eta.txt   自由表面高程
    beta.txt  变底部高程
    EVI.txt   能量/动量时间历程（可选）：每行 t,Ek,Ep,I（逗号分隔，无表头）
文件格式（eta.txt / beta.txt）：
    第 1 行：0, x0, x1, ..., x_{N-1}
    后续行：t_i, f(x0), f(x1), ..., f(x_{N-1})

显示变换：
    η_display = η * eta_scale
    β_display = (β - beta_offset) * beta_scale

默认：
    eta_scale   = 1.0
    beta_offset = β.min()   （把静态基底平移到 0）
    beta_scale  = 1.0
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def eta_hat_kt(x: np.ndarray, eta: np.ndarray):
    """
    对每行 eta[k, :] 做 FFT，返回 (k_axis, eta_hat)。
    k_axis 为角波数 (rad/m)，长度 N，FFTW 顺序（0,+,...,-）。
    eta_hat.shape == eta.shape，复数。
    """
    N = x.size
    dx = x[1] - x[0]
    # 角波数：2π * fftfreq(N, dx)
    k_axis = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    eta_hat = np.fft.fft(eta, axis=1)          # 沿空间轴
    return k_axis, eta_hat

# ---------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------
def load_wide_table(path: Path):
    """
    读取「宽表」格式：
        第 1 行  : 0, x0, x1, ..., x_{N-1}
        第 k 行  : t_k, f(x0), f(x1), ..., f(x_{N-1})
    返回 (x, t, F)，F.shape == (nt, N)。
    """
    data = np.loadtxt(path, delimiter=",")
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError(f"{path}: 至少需要 1 行表头 + 1 行数据")
    x = data[0, 1:]
    t = data[1:, 0]
    F = data[1:, 1:]
    return x, t, F


def load_EVI(path: Path):
    """
    读取 EVI.txt：每行 t,Ek,Ep,I,V（5 列，逗号分隔，无表头）。
    返回 (t, Ek, Ep, I, V)，五者长度相同。
    """
    data = np.loadtxt(path, delimiter=",")
    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError(
            f"{path}: 期望 5 列 (t,Ek,Ep,I,V)，实际 shape={data.shape}")
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]


def padded_range(lo: float, hi: float, pad: float = 0.08):
    span = hi - lo
    if span == 0.0:
        span = max(abs(hi), 1.0)
    return lo - pad * span, hi + pad * span

def build_spectrum_animation(
    eta_path: Path,
    out_path: Path,
    fps: int = 20,
    dpi: int = 150,
    kmin: float | None = None,
    kmax: float | None = None,
    log_scale: bool = False,
    which: str = "abs",          # "abs" | "real" | "imag"
):
    """
    绘制 η 的傅里叶系数 |η̂(k,t)|（或实部/虚部）随时间的演化。
    横轴：角波数 k (rad/m)，只画 k >= 0 的一半。
    """
    x, t, eta = load_wide_table(eta_path)
    nt = len(t)
    N = x.size
    print(f"[spectrum] 读取完成：{nt} 个时间步, {N} 个空间点")

    k_axis, eta_hat = eta_hat_kt(x, eta)

    # 只保留 k >= 0 的频点（排除 FFTW 顺序里末尾的 Nyquist 负频率点）
    pos_mask = k_axis >= 0.0
    k_pos = k_axis[pos_mask]
    hat_pos = eta_hat[:, pos_mask]

    if which == "abs":
        data = np.abs(hat_pos)
        ylabel = r"$|\hat{\eta}(k,t)|$"
    elif which == "real":
        data = hat_pos.real
        ylabel = r"$\mathrm{Re}\,\hat{\eta}(k,t)$"
    elif which == "imag":
        data = hat_pos.imag
        ylabel = r"$\mathrm{Im}\,\hat{\eta}(k,t)$"
    else:
        raise ValueError(f"which 必须是 abs/real/imag，收到 {which}")

    # 横轴范围
    k_lo = 0.0 if kmin is None else kmin
    k_hi = k_pos.max() if kmax is None else kmax
    if k_lo >= k_hi:
        raise ValueError(f"k 轴范围非法：kmin={k_lo}, kmax={k_hi}")

    # 纵轴范围（只看显示窗口内的数据）
    mask = (k_pos >= k_lo) & (k_pos <= k_hi)
    if not np.any(mask):
        raise ValueError("k 轴窗口内没有数据点")
    d_win = data[:, mask]
    if log_scale:
        # log 轴要求正值；加一个小 floor 避免 log(0)
        floor = max(d_win[d_win > 0].min() * 1e-3, 1e-300) if np.any(d_win > 0) else 1e-300
        d_win_plot = np.maximum(d_win, floor)
        y_lo, y_hi = d_win_plot.min() * 0.5, d_win_plot.max() * 2.0
    else:
        y_lo, y_hi = padded_range(float(d_win.min()), float(d_win.max()))
        if which != "abs":
            # 实部/虚部有正负，保证 0 在范围内
            y_lo = min(y_lo, 0.0)
            y_hi = max(y_hi, 0.0)

    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=dpi)
    (line,) = ax.plot([], [], lw=1.6, color="#1f77b4")

    ax.set_xlim(k_lo, k_hi)
    ax.set_ylim(y_lo, y_hi)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("k  (rad/m)")
    ax.set_ylabel(ylabel)
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    
    # 统一留白布局，避免标题被挡
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.08, right=0.97)

    def init():
        line.set_data([], [])
        ax.set_title("Fourier spectrum of η")
        return (line,)

    def update(i: int):
        line.set_data(k_pos, data[i])
        ax.set_title(f"Fourier spectrum of η   t = {t[i]:.4f}   ({i+1}/{nt})")
        return (line,)

    ani = animation.FuncAnimation(
        fig, update, frames=nt, init_func=init,
        interval=1000.0 / fps, blit=False,
    )

    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        ani.save(str(out_path), writer="pillow", fps=fps)
    elif suffix in (".mp4", ".mov", ".mkv", ".webm"):
        ani.save(str(out_path), writer="ffmpeg", fps=fps, dpi=dpi)
    else:
        raise ValueError(f"不支持的输出格式：{suffix}（请用 .mp4 或 .gif）")

    plt.close(fig)
    print(f"[OK] 已生成谱动画：{out_path}")

# ---------------------------------------------------------------------
# 动画主函数
# ---------------------------------------------------------------------
def build_animation(
    eta_path: Path,
    beta_path: Path,
    out_path: Path,
    fps: int = 20,
    dpi: int = 150,
    eta_scale: float = 1.0,
    beta_offset: float | None = None,
    beta_scale: float = 1.0,
    evi_path: Path | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
):
    # ---- 读 eta / beta ----
    x, t, eta = load_wide_table(eta_path)
    x_b, t_b, beta = load_wide_table(beta_path)

    if eta.shape != beta.shape:
        raise ValueError(
            f"eta.txt 与 beta.txt 形状不一致：{eta.shape} vs {beta.shape}"
        )
    if not np.allclose(t, t_b):
        raise ValueError("eta.txt 与 beta.txt 时间列不一致")

    nt = len(t)
    print(f"读取完成：{nt} 个时间步, {len(x)} 个空间点")
    print(f"  t ∈ [{t[0]:.6f}, {t[-1]:.6f}]")

    # ---- 读 EVI（可选）----
    E_total = None
    Ek = Ep = I_mom = V_mom = None
    if evi_path is not None:
        t_evi, Ek, Ep, I_mom, V_mom = load_EVI(evi_path)

        if len(t_evi) != len(t):
            warnings.warn(
                f"EVI.txt 行数({len(t_evi)}) 与 eta.txt 行数({len(t)}) 不一致，"
                "将按时间线性插值到 eta 的时间轴"
            )
            Ek    = np.interp(t, t_evi, Ek)
            Ep    = np.interp(t, t_evi, Ep)
            I_mom = np.interp(t, t_evi, I_mom)
            V_mom = np.interp(t, t_evi, V_mom)
        elif not np.allclose(t, t_evi, rtol=0, atol=1e-12):
            warnings.warn("EVI.txt 时间列与 eta.txt 时间列存在浮点差异")

        E_total = Ek + Ep
        print(f"  已读 EVI.txt："
              f"Ek ∈ [{Ek.min():.6e}, {Ek.max():.6e}], "
              f"Ep ∈ [{Ep.min():.6e}, {Ep.max():.6e}], "
              f"E  ∈ [{E_total.min():.6e}, {E_total.max():.6e}], "
              f"I  ∈ [{I_mom.min():.6e}, {I_mom.max():.6e}], "
              f"V  ∈ [{V_mom.min():.6e}, {V_mom.max():.6e}]")

    # ---- β 的显示变换 ----
    if beta_offset is None:
        beta_offset = float(beta.min())
    beta_disp = (beta - beta_offset) * beta_scale

    # 用于图例/坐标轴显示的 β 基线标签（处理符号，避免出现 "β − -1"）
    if beta_offset < 0:
        beta_label = f"(β + {abs(beta_offset):.6g}) * {beta_scale:.6g}"
    elif beta_offset > 0:
        beta_label = f"(β − {beta_offset:.6g}) * {beta_scale:.6g}"
    else:
        beta_label = f"β * {beta_scale:.6g}"

    # ---- η 的显示变换 ----
    eta_disp = eta * eta_scale
    eta_label = f"η * {eta_scale:.6g}"

    print(f"  η 显示：eta_scale = {eta_scale:.6g}")
    print(f"    η_display 范围 = [{eta_disp.min():.6e}, {eta_disp.max():.6e}]")
    print(f"  β 显示：beta_offset = {beta_offset:.6g}, "
          f"beta_scale = {beta_scale:.6g}")
    print(f"    β_display 范围 = [{beta_disp.min():.6e}, {beta_disp.max():.6e}]")

    # ---- 统一的 Y 轴范围 ----
    y_lo, y_hi = padded_range(
        float(min(eta_disp.min(), beta_disp.min())),
        float(max(eta_disp.max(), beta_disp.max())),
    )

    # ---- 画布与单 Y 轴 ----
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=dpi)

    c_eta  = "#1f77b4"     # 蓝
    c_beta = "#d62728"     # 红

    (line_eta,)  = ax.plot([], [], color=c_eta,  lw=2.3, label=eta_label)
    (line_beta,) = ax.plot([], [], color=c_beta, lw=2.3, label=beta_label)

    x_lo = x.min() if xmin is None else xmin
    x_hi = x.max() if xmax is None else xmax
    if x_lo >= x_hi:
        raise ValueError(f"x 轴范围非法：xmin={x_lo}, xmax={x_hi}")
    ax.set_xlim(x_lo, x_hi)

    ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel("x")
    if eta_scale != 1.0:
        ax.set_ylabel(f"η × {eta_scale:.3e}   /   {beta_label}")
    else:
        ax.set_ylabel(f"η   /   {beta_label}")
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    # 给顶部 info_text 和 ax.set_title 都留足空间
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.08, right=0.97)

    # ---- 图例标题：仅保留 EVI（能量/动量），时间移到全局标题 ----
    def format_legend_info(k: int) -> str:
        if E_total is None:
            return ""  # 没有 EVI 数据时，图例不需要额外标题
        return (f"E  = {E_total[k]:.6e}\n"
                f"Ek = {Ek[k]:.6e}\n"
                f"Ep = {Ep[k]:.6e}\n"
                f"I  = {I_mom[k]:.6e}\n"
                f"V  = {V_mom[k]:.6e}")

    legend_title0 = format_legend_info(0)

    # 如果 legend_title0 为空，Matplotlib 不会显示 title 区域，排版更紧凑
    legend = ax.legend(
        [line_eta, line_beta], [eta_label, beta_label],
        loc="upper right",
        framealpha=0.9,
        title=legend_title0,
        title_fontsize=9,
    )

    # ---- 帧回调 ----
    def init():
        line_eta.set_data([], [])
        line_beta.set_data([], [])
        legend.set_title(format_legend_info(0))
        ax.set_title("Wave / Bathymetry evolution")
        return line_eta, line_beta, legend

    def update(k: int):
        line_eta.set_data(x, eta_disp[k])
        line_beta.set_data(x, beta_disp[k])
        
        # 1. EVI 信息留给图例
        legend.set_title(format_legend_info(k))
        
        # 2. 时间 t 放到上方总标题后面
        ax.set_title(f"Wave / Bathymetry evolution   t = {t[k]:.4f}   ({k+1}/{nt})")
        
        return line_eta, line_beta, legend

    ani = animation.FuncAnimation(
        fig, update,
        frames=nt,
        init_func=init,
        interval=1000.0 / fps,
        blit=False,
    )

    # ---- 保存 ----
    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        ani.save(str(out_path), writer="pillow", fps=fps)
    elif suffix in (".mp4", ".mov", ".mkv", ".webm"):
        ani.save(str(out_path), writer="ffmpeg", fps=fps, dpi=dpi)
    else:
        raise ValueError(f"不支持的输出格式：{suffix}（请用 .mp4 或 .gif）")

    plt.close(fig)
    print(f"[OK] 已生成：{out_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="根据 DNO 输出文件生成 η–x 与 β–x 的演化动画（单 Y 轴，"
                    "可选叠加 EVI 能量/动量显示）"
    )
    ap.add_argument("--eta",  default="eta.txt",
                    help="η 数据文件（默认 eta.txt）")
    ap.add_argument("--beta", default="beta.txt",
                    help="β 数据文件（默认 beta.txt）")
    ap.add_argument("--evi",  default="EVI.txt",
                    help="EVI 数据文件（默认 EVI.txt；传 '' 表示禁用）")
    ap.add_argument("--out",  default="eta_beta_animation.mp4",
                    help="输出文件名；后缀决定格式（.mp4 / .gif）")
    ap.add_argument("--fps",  type=int, default=20,
                    help="帧率，默认 20")
    ap.add_argument("--dpi",  type=int, default=150,
                    help="渲染 dpi，默认 150")
    ap.add_argument("--eta-scale", type=float, default=1.0,
                    help="η 的显示缩放因子（仅用于可视化）。默认 1.0。")
    ap.add_argument("--beta-offset", type=float, default=None,
                    help="β 的显示基线（减法项）。默认取 β 的最小值。")
    ap.add_argument("--beta-scale", type=float, default=1.0,
                    help="β 的显示缩放因子。默认 1.0。")
    ap.add_argument("--xmin", type=float, default=None,
                    help="x 轴显示下限（默认取数据最小值）")
    ap.add_argument("--xmax", type=float, default=None,
                    help="x 轴显示上限（默认取数据最大值）")
    ap.add_argument("--mode", choices=["surface", "spectrum", "both"],
                    default="surface",
                    help="画什么：surface=η–x 动画；spectrum=η̂(k) 动画；both=两者都画")
    ap.add_argument("--spectrum-out", default="eta_spectrum.mp4",
                    help="谱动画输出文件名（默认 eta_spectrum.mp4）")
    ap.add_argument("--which", choices=["abs", "real", "imag"], default="abs",
                    help="谱动画画 |η̂| / Re η̂ / Im η̂（默认 abs）")
    ap.add_argument("--kmin", type=float, default=None,
                    help="谱动画 k 轴下限 (rad/m)，默认 0")
    ap.add_argument("--kmax", type=float, default=None,
                    help="谱动画 k 轴上限 (rad/m)，默认 k 的最大值")
    ap.add_argument("--log-scale", action="store_true",
                    help="谱动画纵轴用对数刻度")
    args = ap.parse_args()

    # 解析 EVI 文件（可选）
    evi: Path | None = None
    if args.evi:                      # 空字符串 → 禁用
        p = Path(args.evi)
        if p.exists():
            evi = p
        else:
            print(f"[warn] 未找到 {p}，将不显示能量信息")
            evi = None

    if args.mode in ("surface", "both"):
        build_animation(
            Path(args.eta), Path(args.beta), Path(args.out),
            fps=args.fps, dpi=args.dpi,
            eta_scale=args.eta_scale,
            beta_offset=args.beta_offset,
            beta_scale=args.beta_scale,
            evi_path=evi,
            xmin=args.xmin, xmax=args.xmax,
        )
    if args.mode in ("spectrum", "both"):
        build_spectrum_animation(
            Path(args.eta), Path(args.spectrum_out),
            fps=args.fps, dpi=args.dpi,
            kmin=args.kmin, kmax=args.kmax,
            log_scale=args.log_scale,
            which=args.which,
        )


if __name__ == "__main__":
    main()
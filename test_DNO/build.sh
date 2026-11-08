#!/usr/bin/env bash
# =====================================================================
# DNO 项目编译 + 运行脚本
#
# 用法：
#   ./build_and_run.sh              # 编译 + 运行
#   ./build_and_run.sh build        # 只编译
#   ./build_and_run.sh run          # 只运行（不编译）
#   ./build_and_run.sh clean        # 清理构建产物
#
# 依赖：g++ (C++17)、FFTW3 (libfftw3)
#   Debian/Ubuntu: sudo apt install build-essential libfftw3-dev
#   RHEL/CentOS:   sudo yum install gcc-c++ fftw-devel
#   macOS:         brew install fftw
# =====================================================================
set -euo pipefail

# ---------- 可配置项 ----------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
BIN="${BUILD_DIR}/dno_main"

# 源文件（按需增删）
SRCS=(
    "${PROJECT_DIR}/var_op_function.cpp"
    "${PROJECT_DIR}/dealiaser.cpp"
    "${PROJECT_DIR}/DNO_calc_OE.cpp"
    "${PROJECT_DIR}/DNO_output_control.cpp"
    "${PROJECT_DIR}/DNO_time_integral.cpp"
    "${PROJECT_DIR}/DNO_time_integral_main.cpp"
)

CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--O2 -std=c++17 -Wall -Wextra -Wpedantic}"
LDFLAGS="${LDFLAGS:-}"
LDLIBS="${LDLIBS:--lfftw3 -lm}"

# ---------- 工具函数 ----------
log()  { printf '\033[1;34m[build]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn ]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

check_deps() {
    command -v "$CXX" >/dev/null 2>&1 \
        || die "找不到编译器 '$CXX'，请安装 g++ 或设置 CXX 环境变量"

    # 检查 fftw3.h
    if ! echo '#include <fftw3.h>' | "$CXX" -x c++ -E - >/dev/null 2>&1; then
        die "找不到 fftw3.h。Debian/Ubuntu: sudo apt install libfftw3-dev；macOS: brew install fftw"
    fi

    # 检查链接库（用一个小测试程序探测）
    local tmp
    tmp="$(mktemp -d)"
    cat > "${tmp}/probe.cpp" <<'EOF'
#include <fftw3.h>
int main() {
    fftw_complex* p = fftw_alloc_complex(1);
    fftw_free(p);
    return 0;
}
EOF
    if ! "$CXX" -std=c++17 "${tmp}/probe.cpp" -o "${tmp}/probe" -lfftw3 -lm >/dev/null 2>&1; then
        rm -rf "$tmp"
        die "无法链接 libfftw3。请确认已安装 FFTW3 开发库，或检查 LDLIBS"
    fi
    rm -rf "$tmp"
}

ensure_sources() {
    local missing=0
    for f in "${SRCS[@]}"; do
        if [[ ! -f "$f" ]]; then
            warn "源文件不存在: $f"
            missing=1
        fi
    done
    [[ $missing -eq 0 ]] || die "存在缺失的源文件，请检查 SRCS 列表"
}

do_clean() {
    log "清理 $BUILD_DIR"
    rm -rf "$BUILD_DIR"
}

do_build() {
    check_deps
    ensure_sources
    mkdir -p "$BUILD_DIR"

    log "编译到 $BIN"
    log "  CXX      = $CXX"
    log "  CXXFLAGS = $CXXFLAGS"
    log "  LDLIBS   = $LDLIBS"

    # shellcheck disable=SC2086
    "$CXX" $CXXFLAGS "${SRCS[@]}" -o "$BIN" $LDFLAGS $LDLIBS

    log "编译成功: $BIN"
}

do_run() {
    [[ -x "$BIN" ]] || die "可执行文件不存在: $BIN（先运行 ./build_and_run.sh build）"
    log "运行 $BIN"
    # 在项目目录下运行，输出文件（eta.txt 等）写在当前目录
    ( cd "$PROJECT_DIR" && "$BIN" )
    log "运行结束"
    log "输出文件（若启用 recorder）："
    for f in eta.txt xi.txt beta.txt EVI.txt; do
        if [[ -f "${PROJECT_DIR}/${f}" ]]; then
            printf '  %-10s %s bytes\n' "$f" "$(stat -c%s "${PROJECT_DIR}/${f}" 2>/dev/null || wc -c < "${PROJECT_DIR}/${f}")"
        fi
    done
}

do_animate() {
    local script="${PROJECT_DIR}/dno_animate.py"
    [[ -f "$script" ]] || die "找不到 $script"
    command -v python3 >/dev/null 2>&1 || die "找不到 python3"

    # 输入/输出全部用绝对路径，避免依赖调用时的工作目录
    local eta="${PROJECT_DIR}/eta.txt"
    local beta="${PROJECT_DIR}/beta.txt"
    local evi="${PROJECT_DIR}/EVI.txt"
    local out_surface="${PROJECT_DIR}/surface.mp4"
    local out_spec="${PROJECT_DIR}/spec.mp4"

    local evi_arg=()
    if [[ -f "$evi" ]]; then
        evi_arg=(--evi "$evi")
    else
        warn "未找到 $evi，动画将不显示能量信息"
        evi_arg=(--evi "")
    fi

    log "生成动画（surface + spectrum）"
    python3 "$script" \
        --eta  "$eta" \
        --beta "$beta" \
        "${evi_arg[@]}" \
        --eta-scale 50 \
        --beta-offset 10 \
        --xmin 0 \
        --xmax 6.2831853 \
        --mode both \
        --out "$out_surface" \
        --spectrum-out "$out_spec" \
        --log-scale
    log "动画已生成: $out_surface"
    log "谱动画已生成: $out_spec"
}

# ---------- 入口 ----------
case "${1:-all}" in
    build)   do_build ;;
    run)     do_run ;;
    animate) do_animate ;;
    clean)   do_clean ;;
    all)     do_build; do_run; do_animate ;;
    *)       die "未知参数 '$1'，可用: build | run | animate | clean | all" ;;
esac
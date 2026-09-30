#!/usr/bin/env python3
"""Generate SYDE 572 Assignment 1 figures and print numerical results."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media"
MEDIA.mkdir(parents=True, exist_ok=True)

INK = "#211A16"
PAPER = "#F6F0E7"
GRID = "#D7CABA"
TERRACOTTA = "#94563B"
BROWN = "#6F5239"
GOLD = "#AD843C"
SLATE = "#49443F"
CLAY = "#784632"
NAVY = "#273D56"
FALL_OLIVE = "#59634B"
FALL_CMAP = LinearSegmentedColormap.from_list(
    "fall", [GOLD, TERRACOTTA, BROWN, NAVY]
)

plt.rcParams.update(
    {
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.family": "DejaVu Serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 160,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.facecolor": PAPER,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ---------------------------------------------------------------------------
# Distance methods
# ---------------------------------------------------------------------------

def dist_sq(x, x0, y0, f):
    return (x - x0) ** 2 + (f(x) - y0) ** 2


def find_distance_newton(x0, y0, f, df, ddf, initial_guess=0.0, tolerance=1e-12, max_iter=100):
    x = float(initial_guess)
    history = [x]
    for _ in range(max_iter):
        D_prime = 2 * (x - x0) + 2 * (f(x) - y0) * df(x)
        D_double_prime = 2 + 2 * (df(x) ** 2) + 2 * (f(x) - y0) * ddf(x)
        if abs(D_double_prime) < 1e-18:
            break
        next_x = x - D_prime / D_double_prime
        history.append(next_x)
        if abs(next_x - x) < tolerance:
            x = next_x
            break
        x = next_x
    d = math.sqrt(max((x - x0) ** 2 + (f(x) - y0) ** 2, 0.0))
    return d, x, history


def golden_section_search(x0, y0, f, a, b, tolerance=1e-12, max_iter=200):
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi
    a, b = float(a), float(b)
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)

    def d2(x):
        return dist_sq(x, x0, y0, f)

    f1, f2 = d2(x1), d2(x2)
    history = [(a, b, x1, x2)]
    for _ in range(max_iter):
        if abs(b - a) <= tolerance:
            break
        if f1 < f2:
            b = x2
            x2, f2 = x1, f1
            x1 = a + resphi * (b - a)
            f1 = d2(x1)
        else:
            a = x1
            x1, f1 = x2, f2
            x2 = b - resphi * (b - a)
            f2 = d2(x2)
        history.append((a, b, x1, x2))
    best_x = 0.5 * (a + b)
    d = math.sqrt(max(d2(best_x), 0.0))
    return d, best_x, history


def distance_to_parabola(x0, y0, a=1.0, b=0.0, c=5.0, initial_guess=None, interval=None):
    """Distance from a point to y = a x^2 + b x + c using both numerical methods."""

    def f(x):
        return a * x * x + b * x + c

    def df(x):
        return 2 * a * x + b

    def ddf(_x):
        return 2 * a

    guess = x0 if initial_guess is None else initial_guess
    newton = find_distance_newton(x0, y0, f, df, ddf, initial_guess=guess)
    lo, hi = (-20.0, 20.0) if interval is None else interval
    golden = golden_section_search(x0, y0, f, lo, hi)
    return {"newton": newton, "golden": golden, "f": f, "df": df, "ddf": ddf}


def analytical_cubic_root(x0, y0=0.0):
    """Unique real root of 2x^3 + (11 - 2 y0) x - x0 = 0 for f(x)=x^2+5."""
    # x^3 + p x + q = 0
    p = (11 - 2 * y0) / 2.0
    q = -x0 / 2.0
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    sqrt_disc = math.sqrt(disc)
    u = -q / 2.0 + sqrt_disc
    v = -q / 2.0 - sqrt_disc
    x = math.copysign(abs(u) ** (1 / 3), u) + math.copysign(abs(v) ** (1 / 3), v)
    return x


# ---------------------------------------------------------------------------
# Part 2: MSE fitting
# ---------------------------------------------------------------------------

POINTS = np.array([[0.0, 0.5], [2.0, 3.5], [1.0, 1.5], [3.0, 7.5]])
XS, YS = POINTS[:, 0], POINTS[:, 1]
N = len(XS)


def mse_line(m, b, xs=XS, ys=YS):
    r = ys - m * xs - b
    return float(np.mean(r**2))


def mse_parabola(a, b, c, xs=XS, ys=YS):
    r = ys - (a * xs**2 + b * xs + c)
    return float(np.mean(r**2))


def analytical_line(xs=XS, ys=YS):
    n = len(xs)
    sx, sy = xs.sum(), ys.sum()
    sxx, sxy = (xs**2).sum(), (xs * ys).sum()
    A = np.array([[sxx, sx], [sx, n]], dtype=float)
    rhs = np.array([sxy, sy], dtype=float)
    m, b = np.linalg.solve(A, rhs)
    return float(m), float(b)


def analytical_parabola(xs=XS, ys=YS):
    n = len(xs)
    x1, x2, x3, x4 = xs.sum(), (xs**2).sum(), (xs**3).sum(), (xs**4).sum()
    y1 = ys.sum()
    xy = (xs * ys).sum()
    x2y = (xs**2 * ys).sum()
    A = np.array(
        [
            [x4, x3, x2],
            [x3, x2, x1],
            [x2, x1, n],
        ],
        dtype=float,
    )
    rhs = np.array([x2y, xy, y1], dtype=float)
    a, b, c = np.linalg.solve(A, rhs)
    return float(a), float(b), float(c)


def newton_coordinate_line(m0=0.0, b0=0.0, n_cycles=8):
    """Alternate 1D Newton steps on m then b. Quadratic MSE => one step is exact per coord."""
    m, b = float(m0), float(b0)
    hist = [(m, b, mse_line(m, b))]
    n = N
    for _ in range(n_cycles):
        # m-step, b fixed
        r = YS - m * XS - b
        g_m = -2.0 / n * np.sum(XS * r)
        h_m = 2.0 / n * np.sum(XS**2)
        m = m - g_m / h_m
        hist.append((m, b, mse_line(m, b)))
        # b-step, m fixed
        r = YS - m * XS - b
        g_b = -2.0 / n * np.sum(r)
        h_b = 2.0
        b = b - g_b / h_b
        hist.append((m, b, mse_line(m, b)))
    return hist


def newton_coordinate_parabola(a0=0.0, b0=0.0, c0=0.0, n_cycles=12):
    a, b, c = float(a0), float(b0), float(c0)
    hist = [(a, b, c, mse_parabola(a, b, c))]
    n = N
    x2 = XS**2
    for _ in range(n_cycles):
        r = YS - (a * x2 + b * XS + c)
        g_a = -2.0 / n * np.sum(x2 * r)
        h_a = 2.0 / n * np.sum(x2**2)
        a = a - g_a / h_a
        hist.append((a, b, c, mse_parabola(a, b, c)))

        r = YS - (a * x2 + b * XS + c)
        g_b = -2.0 / n * np.sum(XS * r)
        h_b = 2.0 / n * np.sum(XS**2)
        b = b - g_b / h_b
        hist.append((a, b, c, mse_parabola(a, b, c)))

        r = YS - (a * x2 + b * XS + c)
        g_c = -2.0 / n * np.sum(r)
        h_c = 2.0
        c = c - g_c / h_c
        hist.append((a, b, c, mse_parabola(a, b, c)))
    return hist


def save(fig, name):
    path = MEDIA / name
    fig.savefig(path, pad_inches=0.18)
    plt.close(fig)
    return path


def annotate_point(ax, x, y, text, color=INK, offset=(8, 8)):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=9,
        color=color,
        fontfamily="DejaVu Sans",
    )


# ---------------------------------------------------------------------------
# Figures — Part 1
# ---------------------------------------------------------------------------

def fig_overview(results):
    fig, ax = plt.subplots(figsize=(8.6, 6.3))
    xs = np.linspace(-10, 8, 500)
    ax.plot(xs, xs**2 + 5, color=BROWN, lw=2.6, label=r"$y = x^2 + 5$")
    colors = [TERRACOTTA, GOLD, NAVY, CLAY, FALL_OLIVE]
    for (x0, y0), rec, col in zip(QUERY_POINTS, results, colors):
        xstar = rec["x_analytic"]
        ystar = xstar**2 + 5
        d = rec["d_analytic"]
        ax.plot(
            [x0, xstar],
            [y0, ystar],
            color=col,
            lw=1.7,
            ls="--",
            label=f"({x0:g}, {y0:g})  →  d = {d:.4f}",
        )
        ax.scatter([x0], [y0], s=64, color=col, zorder=5, edgecolors=INK, linewidths=0.45)
        ax.scatter([xstar], [ystar], s=48, color=col, marker="s", zorder=5, edgecolors=INK, linewidths=0.45)
        ax.annotate(
            f"({x0:g}, {y0:g})",
            (x0, y0),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            fontsize=8,
            color=col,
            fontfamily="DejaVu Sans",
        )
    ax.set_xlim(-10.6, 8.6)
    ax.set_ylim(-2.4, 20)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Shortest distances from query points to $y=x^2+5$")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    save(fig, "p1_overview.png")


def fig_orthogonality():
    x0, y0 = -8.0, 0.0
    xstar = analytical_cubic_root(x0)
    ystar = xstar**2 + 5
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    xs = np.linspace(-10, 2, 400)
    ax.plot(xs, xs**2 + 5, color=BROWN, lw=2.4, label=r"$f(x)=x^2+5$")
    ax.scatter([x0], [y0], s=70, color=TERRACOTTA, zorder=5, label="query $(-8,0)$")
    ax.scatter([xstar], [ystar], s=70, color=GOLD, zorder=5, label="closest point")
    ax.plot([x0, xstar], [y0, ystar], color=TERRACOTTA, lw=1.8)
    # tangent: y - ystar = 2xstar (x - xstar)
    xt = np.linspace(xstar - 2.2, xstar + 2.2, 50)
    yt = ystar + 2 * xstar * (xt - xstar)
    ax.plot(xt, yt, color=NAVY, lw=1.6, label="tangent at closest point")
    ax.annotate(
        r"$(x-x_0)+(f(x)-y_0)f'(x)=0$" + "\n" + "connecting vector ⊥ tangent",
        xy=(xstar, ystar),
        xytext=(-8.8, 9.6),
        ha="left",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=SLATE),
        fontfamily="DejaVu Sans",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Analytical stationarity: orthogonality to the tangent")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.set_xlim(-9.2, 2)
    ax.set_ylim(-1, 11.5)
    ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(bottom=0.24, top=0.88)
    save(fig, "p1_orthogonality.png")


def fig_newton_geometry(x0=-8.0, y0=0.0, guess=-2.5):
    def f(x):
        return x * x + 5

    def df(x):
        return 2 * x

    def ddf(_x):
        return 2.0

    d, xstar, hist = find_distance_newton(x0, y0, f, df, ddf, initial_guess=guess)
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    xs = np.linspace(-10, 2, 400)
    ax.plot(xs, f(xs), color=BROWN, lw=2.4, label=r"$y=x^2+5$")
    ax.scatter([x0], [y0], s=70, color=TERRACOTTA, zorder=6, label="query")
    for i, x in enumerate(hist[:8]):
        if i > 5:
            continue
        y = f(x)
        ax.scatter([x], [y], s=36, color=GOLD, zorder=5)
        ax.plot([x0, x], [y0, y], color=GOLD, lw=0.9, alpha=0.45)
        ax.text(x - 0.12, y + 0.45, f"{i}", fontsize=8, color=CLAY, fontfamily="DejaVu Sans")
    ax.plot([x0, xstar], [y0, f(xstar)], color=TERRACOTTA, lw=2.0, label=f"final d={d:.4f}")
    ax.set_title(f"Newton–Raphson iterates on the curve  (start $x={guess:g}$)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(frameon=False)
    ax.set_xlim(-10.5, 2)
    ax.set_ylim(-1, 22)
    save(fig, "p1_newton_geometry.png")
    return hist, d, xstar


def fig_newton_dprime(x0=-8.0, y0=0.0, guess=-2.5, hist=None):
    def f(x):
        return x * x + 5.0

    def Dp(x):
        return 2 * (x - x0) + 2 * (f(x) - y0) * (2 * x)

    def Dpp(x):
        return 2 + 2 * (2 * x) ** 2 + 2 * (f(x) - y0) * 2

    if hist is None:
        _, _, hist = find_distance_newton(x0, y0, f, lambda x: 2 * x, lambda x: 2.0, guess)
    xs = np.linspace(-5.2, 1.2, 500)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.axhline(0, color=SLATE, lw=0.9)
    ax.plot(xs, [Dp(x) for x in xs], color=BROWN, lw=2.2, label=r"$D'(x)$")
    nshow = min(6, len(hist) - 1)
    for i in range(nshow):
        x = hist[i]
        y = Dp(x)
        slope = Dpp(x)
        xr = np.linspace(x - 0.9, x + 1.2, 30)
        ax.plot(xr, y + slope * (xr - x), color=GOLD, lw=1.05, alpha=0.9)
        ax.scatter([x], [y], color=TERRACOTTA, s=32, zorder=5)
        dy = 14 if y >= -40 else -22
        ax.text(x, y + dy, f"$x_{{{i}}}$", fontsize=8, ha="center")
        if abs(slope) > 1e-12:
            xnext = x - y / slope
            ax.scatter([xnext], [0], color=NAVY, s=18, zorder=6)
            ax.plot([x, xnext], [y, 0], color=NAVY, lw=0.7, ls=":", alpha=0.7)
    ax.set_xlim(-5.2, 1.2)
    ax.set_ylim(-160, 70)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$D'(x)$")
    ax.set_title("Newton steps on $D'(x)=0$ for query $(-8,0)$")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "p1_newton_dprime.png")


def fig_golden(x0=-8.0, y0=0.0, a=-12.0, b=4.0):
    def f(x):
        return x * x + 5

    d, xstar, hist = golden_section_search(x0, y0, f, a, b, tolerance=1e-8)
    xs = np.linspace(a, b, 500)
    Dvals = (xs - x0) ** 2 + (f(xs) - y0) ** 2
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(8.2, 7.0), sharex=True, gridspec_kw={"height_ratios": [1.35, 1.0]}
    )
    ax0.plot(xs, Dvals, color=BROWN, lw=2.2, label=r"$D(x)=(x+8)^2+(x^2+5)^2$")
    ax0.axvline(xstar, color=TERRACOTTA, ls="--", lw=1.2, label=f"minimizer $x^*={xstar:.4f}$")
    ax0.scatter([xstar], [dist_sq(xstar, x0, y0, f)], color=TERRACOTTA, s=40, zorder=5)
    ax0.set_ylim(70, 220)
    ax0.set_ylabel(r"$D(x)$  (zoomed)")
    ax0.set_title("Golden-section search: objective and shrinking bracket")
    ax0.legend(frameon=False, loc="upper right")

    cmap = FALL_CMAP
    nplot = min(18, len(hist))
    for i in range(nplot):
        lo, hi, x1, x2 = hist[i]
        col = cmap(0.25 + 0.7 * i / nplot)
        ax1.plot([lo, hi], [i, i], color=col, lw=3.2, solid_capstyle="round")
        ax1.scatter([x1, x2], [i, i], color=INK, s=10, zorder=5)
    ax1.axvline(xstar, color=TERRACOTTA, ls="--", lw=1.2)
    ax1.set_xlabel("x")
    ax1.set_ylabel("iteration")
    ax1.set_ylim(nplot - 0.8, -0.8)
    ax1.set_yticks(np.arange(0, nplot, 2))
    ax1.set_title("Probe points $u,v$ inside the current interval $[a,b]$", fontsize=12)
    fig.tight_layout()
    save(fig, "p1_golden.png")
    return hist, d, xstar


def fig_all_points_table_plot(results):
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.axis("off")
    cols = ["Point", "Analytic x*", "Analytic d", "Newton d", "Golden d", "Newton iters"]
    cells = []
    for (x0, y0), rec in zip(QUERY_POINTS, results):
        cells.append(
            [
                f"({x0:g}, {y0:g})",
                f"{rec['x_analytic']:.6f}",
                f"{rec['d_analytic']:.6f}",
                f"{rec['d_newton']:.6f}",
                f"{rec['d_golden']:.6f}",
                str(rec["newton_iters"]),
            ]
        )
    table = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor(BROWN)
            cell.set_text_props(color="white", fontfamily="DejaVu Sans")
        else:
            cell.set_facecolor(PAPER if r % 2 else "#EBE1D5")
    ax.set_title("Agreement of analytic, Newton, and golden-section distances", pad=18)
    save(fig, "p1_table.png")


def fig_nonpoly():
    cases = [
        ("exponential", r"$y=e^{x}$", lambda x: np.exp(x), lambda x: np.exp(x), lambda x: np.exp(x), (0.0, 2.0), -1.0, (-3, 2)),
        ("logarithm", r"$y=\ln x$", lambda x: np.log(x), lambda x: 1.0 / x, lambda x: -1.0 / x**2, (3.0, 0.0), 2.0, (0.2, 6)),
        ("denominator", r"$y=\frac{1}{x}$", lambda x: 1.0 / x, lambda x: -1.0 / x**2, lambda x: 2.0 / x**3, (2.0, 2.0), 1.2, (0.3, 4)),
        ("radical", r"$y=\sqrt{x}$", lambda x: np.sqrt(x), lambda x: 0.5 / np.sqrt(x), lambda x: -0.25 / x**1.5, (2.0, 3.0), 2.5, (0.05, 6)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.6))
    extra = []
    for ax, (name, label, f, df, ddf, pt, guess, (lo, hi)) in zip(axes.ravel(), cases):
        x0, y0 = pt
        dN, xN, hist = find_distance_newton(x0, y0, f, df, ddf, initial_guess=guess)
        dG, xG, _ = golden_section_search(x0, y0, f, lo, hi)
        xs = np.linspace(lo, hi, 400)
        ax.plot(xs, f(xs), color=BROWN, lw=2.1, label=label)
        ax.scatter([x0], [y0], color=TERRACOTTA, s=50, zorder=5, label="query")
        ax.scatter([xN], [f(xN)], color=GOLD, s=40, zorder=5)
        ax.plot([x0, xN], [y0, f(xN)], color=TERRACOTTA, lw=1.5, ls="--")
        ax.set_title(f"{name}:  d_N={dN:.4f},  d_G={dG:.4f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(frameon=False, fontsize=8)
        extra.append(
            {
                "name": name,
                "point": pt,
                "newton_x": xN,
                "newton_d": dN,
                "golden_x": xG,
                "golden_d": dG,
                "iters": len(hist) - 1,
            }
        )
    fig.suptitle("Same routines on non-polynomial curves", fontsize=15, y=1.01)
    fig.tight_layout()
    save(fig, "p1_nonpoly.png")
    return extra


def fig_other_parabolas():
    samples = [
        ((1.0, 0.0, 0.0), (3.0, 1.0), r"$y=x^2$"),
        ((0.5, -1.0, 2.0), (-2.0, 4.0), r"$y=0.5x^2-x+2$"),
        ((-0.3, 0.0, 4.0), (1.0, -2.0), r"$y=-0.3x^2+4$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6))
    recs = []
    for ax, ((a, b, c), (x0, y0), lab) in zip(axes, samples):
        out = distance_to_parabola(x0, y0, a, b, c, initial_guess=x0, interval=(-8, 8))
        d, xstar, _ = out["newton"]
        ystar = a * xstar**2 + b * xstar + c
        margin = max(1.0, 0.35 * abs(x0 - xstar))
        x_min = min(x0, xstar) - margin
        x_max = max(x0, xstar) + margin
        xs = np.linspace(x_min, x_max, 300)
        ys = a * xs**2 + b * xs + c
        x_center = 0.5 * (x_min + x_max)
        y_min = min(y0, ystar, float(np.min(ys)))
        y_max = max(y0, ystar, float(np.max(ys)))
        y_center = 0.5 * (y_min + y_max)
        view_size = 1.25 * max(x_max - x_min, y_max - y_min)
        view_x_min = x_center - view_size / 2
        view_x_max = x_center + view_size / 2
        curve_xs = np.linspace(view_x_min, view_x_max, 600)

        ax.plot(curve_xs, a * curve_xs**2 + b * curve_xs + c, color=BROWN, lw=2.0, label=lab)
        ax.scatter([x0], [y0], color=TERRACOTTA, marker="o", s=40, zorder=5)
        ax.scatter([xstar], [ystar], color=GOLD, marker="s", s=36, zorder=5)
        ax.plot([x0, xstar], [y0, ystar], color=TERRACOTTA, ls="--", lw=1.3)
        ax.set_title(f"d={d:.4f}")
        ax.legend(frameon=False, fontsize=7, loc="best")
        ax.set_xlim(view_x_min, view_x_max)
        ax.set_ylim(y_center - view_size / 2, y_center + view_size / 2)
        ax.set_aspect("equal", adjustable="box")
        recs.append({"eq": lab, "point": (x0, y0), "d": d, "x": xstar})
    fig.suptitle("General parabola routine on other $(a,b,c)$ and query points", y=1.03)
    fig.tight_layout()
    save(fig, "p1_other_parabolas.png")
    return recs


def fig_newton_nonpoly_exp():
    x0, y0 = 0.0, 2.0
    f = np.exp
    df = np.exp
    ddf = np.exp
    d, xstar, hist = find_distance_newton(x0, y0, f, df, ddf, initial_guess=-1.0)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    xs = np.linspace(-3, 2, 400)
    ax.plot(xs, np.exp(xs), color=BROWN, lw=2.3, label=r"$y=e^x$")
    ax.scatter([x0], [y0], color=TERRACOTTA, s=60, zorder=5, label="(0, 2)")
    for i, x in enumerate(hist[:8]):
        ax.scatter([x], [np.exp(x)], color=GOLD, s=28, zorder=5)
        ax.text(x, np.exp(x) + 0.12, str(i), fontsize=8)
        ax.plot([x0, x], [y0, np.exp(x)], color=GOLD, alpha=0.35, lw=0.9)
    ax.plot([x0, xstar], [y0, np.exp(xstar)], color=TERRACOTTA, lw=1.8)
    ax.set_title(r"Newton iterates for distance to $e^x$")
    ax.legend(frameon=False)
    save(fig, "p1_newton_exp.png")
    return hist, d, xstar


# ---------------------------------------------------------------------------
# Figures — Part 2
# ---------------------------------------------------------------------------

def fig_fits(m, b, a, bb, c):
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    xr = np.linspace(-0.4, 3.4, 200)
    ax.scatter(XS, YS, s=70, color=INK, zorder=5, label="data")
    for x, y in POINTS:
        ax.annotate(f"({x:g},{y:g})", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8, fontfamily="DejaVu Sans")
    ax.plot(xr, m * xr + b, color=TERRACOTTA, lw=2.2, label=fr"line  $y={m:.3f}x{b:+.3f}$")
    ax.plot(xr, a * xr**2 + bb * xr + c, color=BROWN, lw=2.2, label=fr"parabola  $y={a:.3f}x^2{bb:+.3f}x{c:+.3f}$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Least-squares line and parabola")
    ax.legend(frameon=False)
    save(fig, "p2_fits.png")


def fig_line_contour(hist):
    m_star, b_star = analytical_line()
    ms = np.linspace(0.4, 3.6, 180)
    bs = np.linspace(-2.2, 1.8, 180)
    M, B = np.meshgrid(ms, bs)
    Z = np.mean((YS[None, None, :] - M[..., None] * XS - B[..., None]) ** 2, axis=2)
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    cs = ax.contour(M, B, Z, levels=18, colors=BROWN, linewidths=0.9)
    ax.clabel(cs, fmt="%.2f", fontsize=7)
    path = np.array([[h[0], h[1]] for h in hist[:30]])
    ax.plot(path[:, 0], path[:, 1], color=TERRACOTTA, lw=1.6, marker="o", ms=5, label="coordinate Newton")
    for i, (mm, bb) in enumerate(path[:8]):
        ax.text(mm + 0.04, bb + 0.04, str(i), fontsize=8, color=CLAY)
    ax.scatter([m_star], [b_star], marker="*", s=160, color=GOLD, zorder=6, label="analytic min")
    ax.set_xlabel("slope $m$")
    ax.set_ylabel("intercept $b$")
    ax.set_title(r"Line MSE contours  $\mathrm{MSE}(m,b)$  with Newton path")
    ax.legend(frameon=False)
    save(fig, "p2_line_contour.png")


def fig_line_intermediates(hist):
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    xr = np.linspace(-0.3, 3.3, 100)
    ax.scatter(XS, YS, s=64, color=INK, zorder=5, label="data")
    picks = [0, 1, 2, 3, 4]
    cmap = FALL_CMAP
    for k, i in enumerate(picks):
        if i >= len(hist):
            continue
        m, b, mse = hist[i]
        ax.plot(xr, m * xr + b, color=cmap(0.35 + 0.5 * k / len(picks)), lw=1.8, label=f"step {i}: MSE={mse:.3f}")
    ax.set_title("Line fit: intermediate coordinate-Newton models")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    save(fig, "p2_line_steps.png")


def fig_parabola_intermediates(hist):
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    xr = np.linspace(-0.3, 3.3, 120)
    ax.scatter(XS, YS, s=64, color=INK, zorder=5, label="data")
    picks = [0, 1, 3, 9, 21, 60]
    cmap = FALL_CMAP
    for k, i in enumerate(picks):
        if i >= len(hist):
            continue
        a, b, c, mse = hist[i]
        ax.plot(
            xr,
            a * xr**2 + b * xr + c,
            color=cmap(0.3 + 0.55 * k / len(picks)),
            lw=1.7,
            label=f"step {i}: MSE={mse:.3f}",
        )
    ax.set_title("Parabola fit: intermediate coordinate-Newton models")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    save(fig, "p2_parabola_steps.png")


def fig_mse_history(line_hist, para_hist):
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.semilogy([h[2] for h in line_hist], color=TERRACOTTA, marker="o", lw=1.6, label="line MSE")
    ax.semilogy([h[3] for h in para_hist], color=BROWN, marker="s", lw=1.6, label="parabola MSE")
    ax.set_xlabel("coordinate-Newton step")
    ax.set_ylabel("MSE")
    ax.set_title("MSE vs. iteration (alternating one-parameter Newton)")
    ax.legend(frameon=False)
    save(fig, "p2_mse_history.png")


def fig_residuals(m, b, a, bb, c):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True)
    yhat_l = m * XS + b
    yhat_p = a * XS**2 + bb * XS + c
    axes[0].axhline(0, color=SLATE, lw=0.8)
    axes[0].stem(XS, YS - yhat_l, linefmt=TERRACOTTA, markerfmt="o", basefmt="none")
    axes[0].set_title(f"line residuals  MSE={mse_line(m,b):.4f}")
    axes[1].axhline(0, color=SLATE, lw=0.8)
    axes[1].stem(XS, YS - yhat_p, linefmt=BROWN, markerfmt="s", basefmt="none")
    axes[1].set_title(f"parabola residuals  MSE={mse_parabola(a,bb,c):.4f}")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("residual")
    fig.tight_layout()
    save(fig, "p2_residuals.png")


QUERY_POINTS = [(0.0, 0.0), (-4.0, 0.0), (-8.0, 0.0), (2.0, 0.0), (6.0, 0.0)]


def main():
    results = []
    print("=" * 72)
    print("PART 1  f(x) = x^2 + 5")
    print("=" * 72)
    for x0, y0 in QUERY_POINTS:
        xa = analytical_cubic_root(x0, y0)
        da = math.sqrt((xa - x0) ** 2 + ((xa**2 + 5) - y0) ** 2)
        out = distance_to_parabola(x0, y0, 1.0, 0.0, 5.0, initial_guess=x0 * 0.5, interval=(x0 - 15, x0 + 15))
        dn, xn, hn = out["newton"]
        dg, xg, hg = out["golden"]
        rec = {
            "point": [x0, y0],
            "x_analytic": xa,
            "d_analytic": da,
            "x_newton": xn,
            "d_newton": dn,
            "newton_iters": len(hn) - 1,
            "newton_hist": hn[:12],
            "x_golden": xg,
            "d_golden": dg,
            "golden_iters": len(hg) - 1,
        }
        results.append(rec)
        print(
            f"point ({x0:g},{y0:g}):  x*={xa:.10f}  d={da:.10f}  "
            f"Newton d={dn:.10f} ({rec['newton_iters']} it)  "
            f"Golden d={dg:.10f} ({rec['golden_iters']} it)"
        )

    fig_overview(results)
    fig_orthogonality()
    nhist, _, _ = fig_newton_geometry()
    fig_newton_dprime(hist=nhist)
    fig_golden()
    fig_all_points_table_plot(results)
    extra_fn = fig_nonpoly()
    extra_par = fig_other_parabolas()
    fig_newton_nonpoly_exp()

    print("\nNon-polynomial:")
    for e in extra_fn:
        print(e)
    print("\nOther parabolas:")
    for e in extra_par:
        print(e)

    print("\n" + "=" * 72)
    print("PART 2  fits")
    print("=" * 72)
    m, b = analytical_line()
    a, bb, c = analytical_parabola()
    print(f"line analytic:      m={m:.12f}  b={b:.12f}  MSE={mse_line(m,b):.12f}")
    print(f"parabola analytic:  a={a:.12f}  b={bb:.12f}  c={c:.12f}  MSE={mse_parabola(a,bb,c):.12f}")

    line_hist = newton_coordinate_line(0.0, 0.0, n_cycles=40)
    para_hist = newton_coordinate_parabola(0.0, 0.0, 0.0, n_cycles=80)
    print("line Newton final:", line_hist[-1])
    print("parabola Newton final:", para_hist[-1])
    print("line hist:")
    for i, h in enumerate(line_hist):
        print(f"  {i:02d}  m={h[0]:.8f}  b={h[1]:.8f}  MSE={h[2]:.8f}")
    print("parabola hist (every step, first 30):")
    for i, h in enumerate(para_hist[:30]):
        print(f"  {i:02d}  a={h[0]:.8f}  b={h[1]:.8f}  c={h[2]:.8f}  MSE={h[3]:.8f}")

    fig_fits(m, b, a, bb, c)
    fig_line_contour(line_hist)
    fig_line_intermediates(line_hist)
    fig_parabola_intermediates(para_hist)
    fig_mse_history(line_hist, para_hist)
    fig_residuals(m, b, a, bb, c)

    def py(obj):
        if isinstance(obj, dict):
            return {k: py(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [py(v) for v in obj]
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    payload = {
        "part1": results,
        "nonpoly": extra_fn,
        "other_parabolas": extra_par,
        "line": {"m": m, "b": b, "mse": mse_line(m, b), "hist": line_hist[:24]},
        "parabola": {"a": a, "b": bb, "c": c, "mse": mse_parabola(a, bb, c), "hist": para_hist[:40]},
        "line_newton_final": line_hist[-1],
        "parabola_newton_final": para_hist[-1],
    }
    (ROOT / "code" / "results.json").write_text(json.dumps(py(payload), indent=2))
    print("\nWrote figures to", MEDIA)


if __name__ == "__main__":
    main()

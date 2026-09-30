"""
Least-squares line and parabola fits by
  (1) the normal equations (analytical)
  (2) multivariate Newton–Raphson, one coordinate at a time.

Data: (0, 0.5), (2, 3.5), (1, 1.5), (3, 7.5)
Line:      y = m x + b
Parabola:  y = a x^2 + b x + c
MSE:       (1/n) sum (y_i - yhat_i)^2
"""

from __future__ import annotations

import math


POINTS = [(0.0, 0.5), (2.0, 3.5), (1.0, 1.5), (3.0, 7.5)]
XS = [p[0] for p in POINTS]
YS = [p[1] for p in POINTS]
N = len(POINTS)


def mse_line(m, b):
    return sum((y - m * x - b) ** 2 for x, y in zip(XS, YS)) / N


def mse_parabola(a, b, c):
    return sum((y - (a * x * x + b * x + c)) ** 2 for x, y in zip(XS, YS)) / N


def analytical_line():
    sx = sum(XS)
    sy = sum(YS)
    sxx = sum(x * x for x in XS)
    sxy = sum(x * y for x, y in zip(XS, YS))
    # [ sxx  sx ] [m] = [sxy]
    # [ sx    n ] [b]   [sy ]
    det = sxx * N - sx * sx
    m = (sxy * N - sx * sy) / det
    b = (sxx * sy - sx * sxy) / det
    return m, b


def analytical_parabola():
    x1 = sum(XS)
    x2 = sum(x * x for x in XS)
    x3 = sum(x**3 for x in XS)
    x4 = sum(x**4 for x in XS)
    y1 = sum(YS)
    xy = sum(x * y for x, y in zip(XS, YS))
    x2y = sum(x * x * y for x, y in zip(XS, YS))
    # 3x3 normal equations for [a, b, c]
    A = [
        [x4, x3, x2],
        [x3, x2, x1],
        [x2, x1, N],
    ]
    rhs = [x2y, xy, y1]
    return _solve3(A, rhs)


def _solve3(A, rhs):
    """Cramer's rule for a 3x3 (kept elementary on purpose)."""

    def det(M):
        return (
            M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
            - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
        )

    D = det(A)
    cols = list(zip(*A))
    out = []
    for j in range(3):
        B = [list(row) for row in A]
        for i in range(3):
            B[i][j] = rhs[i]
        out.append(det(B) / D)
    return tuple(out)


def newton_coordinate_line(m0=0.0, b0=0.0, n_cycles=8):
    """
    One-parameter Newton, alternating m then b.
    MSE is quadratic, so each 1D Newton step is exact for that coordinate.
    """
    m, b = float(m0), float(b0)
    hist = [(m, b, mse_line(m, b))]
    for _ in range(n_cycles):
        r = [y - m * x - b for x, y in zip(XS, YS)]
        g_m = -2.0 / N * sum(x * ri for x, ri in zip(XS, r))
        h_m = 2.0 / N * sum(x * x for x in XS)
        m = m - g_m / h_m
        hist.append((m, b, mse_line(m, b)))

        r = [y - m * x - b for x, y in zip(XS, YS)]
        g_b = -2.0 / N * sum(r)
        h_b = 2.0
        b = b - g_b / h_b
        hist.append((m, b, mse_line(m, b)))
    return hist


def newton_coordinate_parabola(a0=0.0, b0=0.0, c0=0.0, n_cycles=20):
    a, b, c = float(a0), float(b0), float(c0)
    hist = [(a, b, c, mse_parabola(a, b, c))]
    x2 = [x * x for x in XS]
    for _ in range(n_cycles):
        r = [y - (a * xx + b * x + c) for x, xx, y in zip(XS, x2, YS)]
        g_a = -2.0 / N * sum(xx * ri for xx, ri in zip(x2, r))
        h_a = 2.0 / N * sum(xx * xx for xx in x2)
        a = a - g_a / h_a
        hist.append((a, b, c, mse_parabola(a, b, c)))

        r = [y - (a * xx + b * x + c) for x, xx, y in zip(XS, x2, YS)]
        g_b = -2.0 / N * sum(x * ri for x, ri in zip(XS, r))
        h_b = 2.0 / N * sum(x * x for x in XS)
        b = b - g_b / h_b
        hist.append((a, b, c, mse_parabola(a, b, c)))

        r = [y - (a * xx + b * x + c) for x, xx, y in zip(XS, x2, YS)]
        g_c = -2.0 / N * sum(r)
        h_c = 2.0
        c = c - g_c / h_c
        hist.append((a, b, c, mse_parabola(a, b, c)))
    return hist


if __name__ == "__main__":
    m, b = analytical_line()
    a, bb, c = analytical_parabola()
    print("Analytical line:     m={:.8f}  b={:.8f}  MSE={:.8f}".format(m, b, mse_line(m, b)))
    print(
        "Analytical parabola: a={:.8f}  b={:.8f}  c={:.8f}  MSE={:.8f}".format(
            a, bb, c, mse_parabola(a, bb, c)
        )
    )
    print("\nCoordinate Newton (line), start (0,0):")
    for i, (mm, bb, mse) in enumerate(newton_coordinate_line()):
        print(f"  {i:02d}  m={mm: .8f}  b={bb: .8f}  MSE={mse:.8f}")
    print("\nCoordinate Newton (parabola), start (0,0,0):")
    for i, (aa, bb, cc, mse) in enumerate(newton_coordinate_parabola()[:24]):
        print(f"  {i:02d}  a={aa: .8f}  b={bb: .8f}  c={cc: .8f}  MSE={mse:.8f}")

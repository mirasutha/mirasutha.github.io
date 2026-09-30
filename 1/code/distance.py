"""
Shortest distance from a point to a curve.

Two numerical methods:
  - Newton–Raphson on D'(x) = 0  (needs f, f', f'')
  - Golden-section search on D(x) (needs only f, plus a bracket [a, b])

The parabola helper wraps both for any y = a x^2 + b x + c.
"""

from __future__ import annotations

import math


def squared_distance(x, x0, y0, f):
    return (x - x0) ** 2 + (f(x) - y0) ** 2


def find_distance_newton(
    x0,
    y0,
    f,
    df,
    ddf,
    initial_guess=0.0,
    tolerance=1e-12,
    max_iter=100,
):
    """Return (distance, x_star, iterate_history)."""
    x = float(initial_guess)
    history = [x]
    for _ in range(max_iter):
        D_prime = 2.0 * (x - x0) + 2.0 * (f(x) - y0) * df(x)
        D_double_prime = 2.0 + 2.0 * (df(x) ** 2) + 2.0 * (f(x) - y0) * ddf(x)
        if abs(D_double_prime) < 1e-18:
            break
        next_x = x - D_prime / D_double_prime
        history.append(next_x)
        if abs(next_x - x) < tolerance:
            x = next_x
            break
        x = next_x
    distance = math.sqrt(max(squared_distance(x, x0, y0, f), 0.0))
    return distance, x, history


def golden_section_search(
    x0,
    y0,
    f,
    a,
    b,
    tolerance=1e-12,
    max_iter=200,
):
    """Minimize D(x) on [a, b]. Return (distance, x_star, bracket_history)."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    resphi = 2.0 - phi
    a, b = float(a), float(b)
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)

    def D(x):
        return squared_distance(x, x0, y0, f)

    f1, f2 = D(x1), D(x2)
    history = [(a, b, x1, x2)]
    for _ in range(max_iter):
        if abs(b - a) <= tolerance:
            break
        if f1 < f2:
            b = x2
            x2, f2 = x1, f1
            x1 = a + resphi * (b - a)
            f1 = D(x1)
        else:
            a = x1
            x1, f1 = x2, f2
            x2 = b - resphi * (b - a)
            f2 = D(x2)
        history.append((a, b, x1, x2))
    best_x = 0.5 * (a + b)
    return math.sqrt(max(D(best_x), 0.0)), best_x, history


def distance_to_parabola(
    x0,
    y0,
    a=1.0,
    b=0.0,
    c=0.0,
    initial_guess=None,
    interval=(-20.0, 20.0),
):
    """Distance from (x0, y0) to the parabola y = a x^2 + b x + c."""

    def f(x):
        return a * x * x + b * x + c

    def df(x):
        return 2.0 * a * x + b

    def ddf(_x):
        return 2.0 * a

    guess = x0 if initial_guess is None else initial_guess
    newton = find_distance_newton(x0, y0, f, df, ddf, initial_guess=guess)
    golden = golden_section_search(x0, y0, f, interval[0], interval[1])
    return newton, golden


if __name__ == "__main__":
    # Assignment parabola y = x^2 + 5 and the five query points.
    points = [(0.0, 0.0), (-4.0, 0.0), (-8.0, 0.0), (2.0, 0.0), (6.0, 0.0)]
    print("Distance to y = x^2 + 5")
    for x0, y0 in points:
        newt, gold = distance_to_parabola(
            x0, y0, a=1, b=0, c=5, initial_guess=0.5 * x0, interval=(x0 - 15, x0 + 15)
        )
        print(
            f"  ({x0:g}, {y0:g})  Newton: d={newt[0]:.8f}, x={newt[1]:.8f}  "
            f"Golden: d={gold[0]:.8f}, x={gold[1]:.8f}"
        )

    # Non-polynomial smoke tests
    import math as _m

    d, x, _ = find_distance_newton(0, 2, _m.exp, _m.exp, _m.exp, initial_guess=-1)
    print(f"exp: d={d:.8f} at x={x:.8f}")
    d, x, _ = golden_section_search(3, 0, _m.log, 0.2, 6)
    print(f"log: d={d:.8f} at x={x:.8f}")

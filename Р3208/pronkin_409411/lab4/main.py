#!/usr/bin/env python3
"""
Лабораторная: аппроксимация таблицы (x, y) методом наименьших квадратов.

Модели
------
1. Линейная           y = a + b·x
2. Квадратичная       y = a + b·x + c·x²
3. Кубическая         y = a + b·x + c·x² + d·x³
4. Экспоненциальная   y = a·e^{b·x}          (требует y>0)
5. Логарифмическая    y = a + b·ln x         (требует x>0)
6. Степенная          y = a·x^{b}            (требует x,y>0)

"""

from __future__ import annotations
import sys
import math
import argparse
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────── вспомогательные структуры ────────────────────── #

@dataclass
class Fit:
    name: str
    f: Callable[[np.ndarray], np.ndarray]  # аппроксимирующая функция
    coeffs: Sequence[float]
    sse: float                             # сумма квадратов ошибок
    rms: float                             # среднеквадратичное отклонение
    r2: float                              # коэффициент детерминации


# ─────────────────────────── построение полиномов ─────────────────────────── #

def _poly_fit(x: np.ndarray, y: np.ndarray, deg: int) -> Fit:
    """Аппроксимация полиномом степени deg (0 ≤ deg ≤ 3) аналитически."""
    # матрица Вандермонда: [1, x, x², …]
    A = np.vander(x, N=deg + 1, increasing=True)  # shape (n, deg+1)
    ATA = A.T @ A
    ATy = A.T @ y
    coeffs = np.linalg.solve(ATA, ATy)  # a0, a1, …
    f = lambda t, c=coeffs: sum(c[i] * t ** i for i in range(len(c)))
    residuals = f(x) - y
    sse = float(np.sum(residuals ** 2))
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    r2 = float(1 - sse / np.sum((y - y.mean()) ** 2))
    names = {1: "Линейная", 2: "Полиномиальная 2‑й ст.", 3: "Полиномиальная 3‑й ст."}
    return Fit(names[deg], f, coeffs, sse, rms, r2)


# ─────────────────────────── нелинейные модели ────────────────────────────── #

def _linear_ls(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Простая линейная регрессия a + b·x через формулы МНК."""
    n = x.size
    sx, sy = x.sum(), y.sum()
    sxx = np.dot(x, x)
    sxy = np.dot(x, y)
    denom = n * sxx - sx * sx
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return a, b


def _exp_fit(x: np.ndarray, y: np.ndarray) -> Fit:
    if np.any(y <= 0):
        raise ValueError("y ≤ 0 → экспоненциальная модель неприменима")
    y_ = np.log(y)
    a_, b_ = _linear_ls(x, y_)
    a = math.exp(a_)
    b = b_
    f = lambda t, A=a, B=b: A * np.exp(B * t)
    name = "Экспоненциальная"
    return _evaluate_model(name, f, (a, b), x, y)


def _log_fit(x: np.ndarray, y: np.ndarray) -> Fit:
    if np.any(x <= 0):
        raise ValueError("x ≤ 0 → логарифмическая модель неприменима")
    x_ = np.log(x)
    a, b = _linear_ls(x_, y)
    f = lambda t, A=a, B=b: A + B * np.log(t)
    return _evaluate_model("Логарифмическая", f, (a, b), x, y)


def _power_fit(x: np.ndarray, y: np.ndarray) -> Fit:
    if np.any((x <= 0) | (y <= 0)):
        raise ValueError("x ≤ 0 или y ≤ 0 → степенная модель неприменима")
    x_, y_ = np.log(x), np.log(y)
    a_, b_ = _linear_ls(x_, y_)
    a = math.exp(a_)
    b = b_
    f = lambda t, A=a, B=b: A * t ** B
    return _evaluate_model("Степенная", f, (a, b), x, y)


def _evaluate_model(name: str, f: Callable[[np.ndarray], np.ndarray],
                    coeffs: Sequence[float], x: np.ndarray, y: np.ndarray) -> Fit:
    residuals = f(x) - y
    sse = float(np.sum(residuals ** 2))
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    r2 = float(1 - sse / np.sum((y - y.mean()) ** 2))
    return Fit(name, f, coeffs, sse, rms, r2)


# ─────────────────────────── метрики ──────────────────────────────────────── #

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    xm, ym = x.mean(), y.mean()
    num = np.sum((x - xm) * (y - ym))
    den = np.sqrt(np.sum((x - xm) ** 2) * np.sum((y - ym) ** 2))
    return float(num / den)


# ─────────────────────────── ввод данных ──────────────────────────────────── #

def _read_points(path: str | None) -> tuple[np.ndarray, np.ndarray]:
    if path:
        src = open(path, encoding="utf-8")
    else:
        print("Введите пары x y по одной на строке (пустая строка — конец):")
        src = sys.stdin
    xs, ys = [], []
    for line in src:
        if not line.strip():
            break
        parts = line.replace(",", ".").split()
        if len(parts) != 2:
            print(f"! строка пропущена: {line.strip()}", file=sys.stderr)
            continue
        xs.append(float(parts[0]))
        ys.append(float(parts[1]))
    if path:
        src.close()
    if not (8 <= len(xs) <= 12):
        raise ValueError("Нужно от 8 до 12 точек")
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


# ─────────────────────────── вывод табличных данных ───────────────────────── #

def _table(x: np.ndarray, y: np.ndarray, fit: Fit):
    print(f"\nТаблица для {fit.name}:")
    print(f"{'i':>3} {'x':>10} {'y':>10} {'φ(x)':>10} {'ε':>10}")
    for i, (xi, yi, fi) in enumerate(zip(x, y, fit.f(x)), 1):
        print(f"{i:3d} {xi:10.4g} {yi:10.4g} {fi:10.4g} {fi - yi:10.4g}")


# ─────────────────────────── визуализация ─────────────────────────────────── #

def _plot_all(x: np.ndarray, y: np.ndarray, fits: list[Fit], best: Fit):
    x_range = np.ptp(x)
    x_dense = np.linspace(x.min() - 0.1 * x_range,
                          x.max() + 0.1 * x_range, 500)
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, marker='o', label="Данные", zorder=5)
    for fit in fits:
        plt.plot(x_dense, fit.f(x_dense), label=fit.name)
    plt.title(f"Аппроксимация (лучшая: {best.name})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ─────────────────────────── основная программа ──────────────────────────── #

def main():
    p = argparse.ArgumentParser(description="Аппроксимация МНК без SciPy")
    p.add_argument("-i", "--input", help="Файл с точками (x y)")
    args = p.parse_args()

    x, y = _read_points(args.input)

    fits: list[Fit] = []
    # полиномы
    for deg in (1, 2, 3):
        fits.append(_poly_fit(x, y, deg))
    # нелинейные
    for fn in (_exp_fit, _log_fit, _power_fit):
        try:
            fits.append(fn(x, y))
        except ValueError as e:
            print(f"⚠ {e}", file=sys.stderr)

    # сортировка по RMS
    fits.sort(key=lambda f: f.rms)
    best = fits[0]

    # вывод
    print("===== РЕЗУЛЬТАТЫ =====")
    r_lin = pearson_r(x, y)
    print(f"Коэффициент Пирсона (линейная): r = {r_lin:.5f}")
    for fit in fits:
        coeffs = ", ".join(f"{c:.6g}" for c in fit.coeffs)
        print(f"\n-- {fit.name} --")
        print(f"Коэффициенты: {coeffs}")
        print(f"SSE = {fit.sse:.6g}")
        print(f"RMS = {fit.rms:.6g}")
        print(f"R²  = {fit.r2:.6g}")
        _table(x, y, fit)

    print(f"\nЛучшая модель: {best.name} (RMS = {best.rms:.6g})")

    _plot_all(x, y, fits, best)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

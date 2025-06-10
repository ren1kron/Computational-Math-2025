import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------
# Класс-решатель для системы методом простых итераций
# ---------------------------------------
class SystemFixedPointSolver:
    def __init__(self, phi1, phi2, dphi1_dx, dphi1_dy, dphi2_dx, dphi2_dy, x0: float, y0: float, tol: float,
                 max_iter: int = 1000):
        """
        Инициализация решателя системы методом простых итераций.
        Параметры:
          phi1, phi2 - функции итерационного преобразования
          dphi1_dx, dphi1_dy, dphi2_dx, dphi2_dy - частные производные (аналитические функции)
          x0, y0 - начальные приближения
          tol - требуемая погрешность
          max_iter - максимальное число итераций
        """
        self.phi1 = phi1
        self.phi2 = phi2
        self.dphi1_dx = dphi1_dx
        self.dphi1_dy = dphi1_dy
        self.dphi2_dx = dphi2_dx
        self.dphi2_dy = dphi2_dy
        self.x0 = x0
        self.y0 = y0
        self.tol = tol
        self.max_iter = max_iter

    def check_convergence_condition(self):
        """
        Проверяет достаточное условие сходимости:
        Вычисляется норма якобиана F(x,y) = (phi1, phi2) в начальной точке,
        где норма оценивается как max{ |dphi1/dx|+|dphi1/dy|, |dphi2/dx|+|dphi2/dy| }.
        Возвращает (True, norm) если условие выполнено (norm < 1).
        """
        x, y = self.x0, self.y0
        J1 = np.abs(self.dphi1_dx(x, y)) + np.abs(self.dphi1_dy(x, y))
        J2 = np.abs(self.dphi2_dx(x, y)) + np.abs(self.dphi2_dy(x, y))
        norm = max(J1, J2)
        return (norm < 1), norm

    def solve(self):
        """
        Выполняет итерационный процесс, пока разница между соседними приближениями
        не станет меньше tol по обеим координатам.
        Возвращает кортеж:
           ((x, y), [|Δx|, |Δy|], итераций)
        """
        x_old, y_old = self.x0, self.y0
        iterations = 0
        while iterations < self.max_iter:
            iterations += 1
            x_new = self.phi1(x_old, y_old)
            y_new = self.phi2(x_old, y_old)
            err_x = np.abs(x_new - x_old)
            err_y = np.abs(y_new - y_old)
            if max(err_x, err_y) < self.tol:
                return (x_new, y_new), [err_x, err_y], iterations
            x_old, y_old = x_new, y_new
        raise RuntimeError("Метод не сошелся за заданное число итераций.")


# ---------------------------------------
# Функция для построения графика контуров
# ---------------------------------------
def plot_system(system_funcs, solution, x_range: tuple, y_range: tuple):
    """
    Строит нулевые контуры двух функций системы и отмечает найденное решение.
    system_funcs - кортеж (F1, F2), где F1(x,y)=0 и F2(x,y)=0.
    """
    F1, F2 = system_funcs
    X, Y = np.meshgrid(np.linspace(x_range[0], x_range[1], 400),
                       np.linspace(y_range[0], y_range[1], 400))
    Z1 = F1(X, Y)
    Z2 = F2(X, Y)

    plt.figure()
    cp1 = plt.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2)
    cp2 = plt.contour(X, Y, Z2, levels=[0], colors='green', linewidths=2)
    plt.clabel(cp1, fmt='F1=0', fontsize=10)
    plt.clabel(cp2, fmt='F2=0', fontsize=10)
    plt.plot(solution[0], solution[1], 'ro', markersize=8, label='Найденное решение')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Нулевые контуры функций системы и найденное решение")
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------------------
# Основная функция
# ---------------------------------------
def main():
    print("Выберите систему нелинейных уравнений:")
    print("1. Система:")
    print("   sin(x+y) = 1.5x - 0.1")
    print("   x^2 + 2y^2 = 1")
    print()
    print("2. Система:")
    print("   tan(x*y + 0.1) = x^2")
    print("   x^2 + 2y^2 = 1")
    system_choice = input("Введите номер системы (1 или 2): ").strip()
    if system_choice not in ["1", "2"]:
        print("Неверный выбор системы.")
        return

    try:
        x0 = float(input("Введите начальное приближение x0: "))
        y0 = float(input("Введите начальное приближение y0: "))
        tol = float(input("Введите требуемую погрешность (например, 0.01): "))
    except ValueError:
        print("Некорректный ввод числовых значений.")
        return

    # Определяем знаки для итерационных преобразований
    s_x = 1 if x0 >= 0 else -1
    s_y = 1 if y0 >= 0 else -1

    # В зависимости от выбора, задаём phi-функции, их производные и функции для построения графика
    if system_choice == "1":
        # Система 1: sin(x+y) = 1.5x - 0.1,  x^2 + 2y^2 = 1
        # Итерационные преобразования:
        phi1 = lambda x, y: (np.sin(x + y) + 0.1) / 1.5
        phi2 = lambda x, y: s_y * np.sqrt((1 - x ** 2) / 2)
        # Производные:
        dphi1_dx = lambda x, y: np.cos(x + y) / 1.5
        dphi1_dy = lambda x, y: np.cos(x + y) / 1.5
        dphi2_dx = lambda x, y: - s_y * x / (np.sqrt(2) * np.sqrt(1 - x ** 2))
        dphi2_dy = lambda x, y: 0
        # Функции для графика (нулевые контуры):
        F1 = lambda x, y: np.sin(x + y) - (1.5 * x - 0.1)
        F2 = lambda x, y: x ** 2 + 2 * y ** 2 - 1
        system_funcs = (F1, F2)
    else:
        # Система 2: tan(x*y+0.1) = x^2,  x^2+2y^2=1
        # Итерационные преобразования:
        # Для phi1: x = s_x*sqrt(tan(x*y+0.1))
        phi1 = lambda x, y: s_x * np.sqrt(np.tan(x * y + 0.1)) if np.tan(x * y + 0.1) >= 0 else \
            (_ for _ in ()).throw(ValueError("tan(x*y+0.1) получило отрицательное значение"))
        phi2 = lambda x, y: s_y * np.sqrt((1 - x ** 2) / 2)
        # Производные для phi1:
        # d/dx sqrt(tan(x*y+0.1)) = (1/(2*sqrt(tan(x*y+0.1)))) * sec^2(x*y+0.1)*y
        dphi1_dx = lambda x, y: s_x * (y * (1 / np.cos(x * y + 0.1) ** 2) / (2 * np.sqrt(np.tan(x * y + 0.1))))
        dphi1_dy = lambda x, y: s_x * (x * (1 / np.cos(x * y + 0.1) ** 2) / (2 * np.sqrt(np.tan(x * y + 0.1))))
        dphi2_dx = lambda x, y: - s_y * x / (np.sqrt(2) * np.sqrt(1 - x ** 2))
        dphi2_dy = lambda x, y: 0
        # Функции для графика:
        F1 = lambda x, y: np.tan(x * y + 0.1) - x ** 2
        F2 = lambda x, y: x ** 2 + 2 * y ** 2 - 1
        system_funcs = (F1, F2)

    # Создаём экземпляр решателя
    solver = SystemFixedPointSolver(phi1, phi2, dphi1_dx, dphi1_dy, dphi2_dx, dphi2_dy, x0, y0, tol)

    # Проверка достаточного условия сходимости
    convergent, norm_val = solver.check_convergence_condition()
    if not convergent:
        print("Достаточное условие сходимости не выполнено.")
        print("Норма якобиана в начальной точке равна {:.4f} (должна быть < 1).".format(norm_val))
        return
    else:
        print("Достаточное условие сходимости выполнено (норма якобиана = {:.4f}).".format(norm_val))

    # Решаем систему
    try:
        (x_sol, y_sol), errors, iterations = solver.solve()
    except RuntimeError as e:
        print("Ошибка:", e)
        return
    except ValueError as ve:
        print("Ошибка в вычислениях:", ve)
        return

    # Вывод результатов
    print("\nНайденное решение системы:")
    print("  x = {:.8f}".format(x_sol))
    print("  y = {:.8f}".format(y_sol))
    print("Количество итераций:", iterations)
    print("Вектор погрешностей последней итерации: |Δx| = {:.8e},  |Δy| = {:.8e}".format(errors[0], errors[1]))

    # Подстановка найденного решения в исходную систему для проверки.
    # Вычисляем значения функций F1 и F2 (для выбранной системы)
    val1 = system_funcs[0](x_sol, y_sol)
    val2 = system_funcs[1](x_sol, y_sol)
    print("\nПогрешности подстановки (значения функций в решении):")
    print("  F1(x, y) = {:.8e}".format(val1))
    print("  F2(x, y) = {:.8e}".format(val2))

    # Определяем диапазоны для графика (с учетом найденного решения)
    x_range = (x_sol - 1, x_sol + 1)
    y_range = (y_sol - 1, y_sol + 1)
    plot_system(system_funcs, (x_sol, y_sol), x_range, y_range)


if __name__ == "__main__":
    main()

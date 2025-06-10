import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Определение функций и их производных
# ---------------------------

# Функция 1: -1.8x^3 - 2.94x^2 + 10.37x + 5.38
def f1(x):
    return -1.8 * x ** 3 - 2.94 * x ** 2 + 10.37 * x + 5.38


def df1(x):
    return -5.4 * x ** 2 - 5.88 * x + 10.37


def f1_2(x):
    return -10.8 * x - 5.88


# Функция 2: x^3 + 2.84x^2 - 5.606x - 14.766
def f2(x):
    return x ** 3 + 2.84 * x ** 2 - 5.606 * x - 14.766


def df2(x):
    return 3 * x ** 2 + 5.68 * x - 5.606


def f2_2(x):
    return 6 * x + 5.68


# Функция 3: -1.38x^3 - 5.42x^2 + 2.57x + 10.95
def f3(x):
    return -1.38 * x ** 3 - 5.42 * x ** 2 + 2.57 * x + 10.95


def df3(x):
    return -4.14 * x ** 2 - 10.84 * x + 2.57


def f3_2(x):
    return -8.28 * x - 10.84


# Функция 4 (трансцендентная): cos(x) - x

def f4(x):
    return np.cos(x) - x


def df4(x):
    return -np.sin(x) - 1


def f4_2(x):
    return -np.cos(x)


# Словарь с функциями для выбора
functions = {
    "1": {
        "name": "f1(x) = -1.8x^3 - 2.94x^2 + 10.37x + 5.38",
        "f": f1,
        "df": df1,
        "f2": f1_2
    },
    "2": {
        "name": "f2(x) = x^3 + 2.84x^2 - 5.606x - 14.766",
        "f": f2,
        "df": df2,
        "f2": f2_2
    },
    "3": {
        "name": "f3(x) = -1.38x^3 - 5.42x^2 + 2.57x + 10.95",
        "f": f3,
        "df": df3,
        "f2": f3_2
    },
    "4": {
        "name": "f4(x) = cos(x) - x",
        "f": f4,
        "df": df4,
        "f2": f4_2
    }
}


# ---------------------------
# Классы-решатели для методов
# ---------------------------
class FixedPointIterationSolver:
    def __init__(self, f, df, a: float, b: float, eps: float):
        """
        Инициализация решателя методом простой итерации.
        Параметры:
          f - функция, корень которой ищем
          df - её производная
          a, b - границы интервала (при a > b они меняются местами)
          eps - требуемая точность
        """
        self.f = f
        self.df = df
        self.eps = eps
        self.a, self.b = (a, b) if a <= b else (b, a)
        self.lambda_val = None

    def count_sign_changes(self, num_points=1000) -> int:
        xs = np.linspace(self.a, self.b, num_points)
        f_vals = self.f(xs)
        signs = np.sign(f_vals)
        changes = 0
        for i in range(1, len(signs)):
            if signs[i] == 0 or signs[i - 1] == 0:
                continue
            if signs[i] != signs[i - 1]:
                changes += 1
        return changes

    def check_root_existence(self) -> None:
        sign_changes = self.count_sign_changes()
        if sign_changes != 1:
            raise ValueError(
                f"На интервале [{self.a}, {self.b}] обнаружено {sign_changes} смен(ы) знака, что не гарантирует наличие ровно одного корня."
            )

    def compute_lambda(self) -> None:
        xs = np.linspace(self.a, self.b, 1000)
        df_vals = self.df(xs)
        m = np.min(df_vals)
        M = np.max(df_vals)
        if m * M < 0:
            raise ValueError("Достаточное условие сходимости не выполнено: производная меняет знак на интервале.")
        self.lambda_val = 2 / (m + M)
        if not np.all(np.abs(1 - self.lambda_val * self.df(xs)) < 1):
            raise ValueError("Достаточное условие сходимости не выполнено для выбранного интервала и lambda.")

    def choose_initial_approximation(self) -> float:
        return self.a if abs(self.f(self.a)) < abs(self.f(self.b)) else self.b

    def solve(self) -> (float, float, int):
        """
        Выполняет итерационный процесс до достижения заданной точности.
        Возвращает кортеж: (найденный корень, значение функции в корне, число итераций).
        """
        self.check_root_existence()
        self.compute_lambda()
        x_prev = self.choose_initial_approximation()
        iteration = 0

        while True:
            x_next = x_prev - self.lambda_val * self.f(x_prev)
            iteration += 1
            if abs(x_next - x_prev) < self.eps:
                break
            x_prev = x_next
            if iteration > 10000:
                raise RuntimeError("Превышено максимальное число итераций.")
        return x_next, self.f(x_next), iteration


class ChordMethodSolver:
    def __init__(self, f, f2, a: float, b: float, eps: float):
        """
        Инициализация решателя методом хорд с фиксированным концом.
        Параметры:
          f - функция
          f2 - вторая производная функции f (для выбора фиксированного конца)
          a, b - границы интервала (при a > b они меняются местами)
          eps - требуемая точность
        """
        self.f = f
        self.f2 = f2
        self.eps = eps
        self.a, self.b = (a, b) if a <= b else (b, a)
        self.fixed_endpoint = None
        self.variable_endpoint = None

    def count_sign_changes(self, num_points=1000) -> int:
        xs = np.linspace(self.a, self.b, num_points)
        f_vals = self.f(xs)
        signs = np.sign(f_vals)
        changes = 0
        for i in range(1, len(signs)):
            if signs[i] == 0 or signs[i - 1] == 0:
                continue
            if signs[i] != signs[i - 1]:
                changes += 1
        return changes

    def check_root_existence(self) -> None:
        sign_changes = self.count_sign_changes()
        if sign_changes != 1:
            raise ValueError(
                f"На интервале [{self.a}, {self.b}] обнаружено {sign_changes} смен(ы) знака, что не гарантирует наличие ровно одного корня."
            )

    def choose_fixed_endpoint(self) -> None:
        """
        Выбирает фиксированный конец по условию: f(x)*f''(x) > 0.
        Если условие выполняется для a, фиксированный конец a, переменная – b; иначе, аналогично для b.
        """
        if self.f(self.a) * self.f2(self.a) > 0:
            self.fixed_endpoint = self.a
            self.variable_endpoint = self.b
        elif self.f(self.b) * self.f2(self.b) > 0:
            self.fixed_endpoint = self.b
            self.variable_endpoint = self.a
        else:
            raise ValueError(
                "Невозможно выбрать фиксированный конец, не удовлетворены условия f(x)*f''(x) > 0 на концах интервала.")

    def solve(self) -> (float, float, int):
        """
        Выполняет итерационный процесс по схеме метода хорд с фиксированным концом:
          x_(n+1) = x_n - (x_n - c)*f(x_n)/(f(x_n)-f(c))
        где c – выбранный фиксированный конец.
        Возвращает кортеж: (найденный корень, значение функции в корне, число итераций).
        """
        self.check_root_existence()
        self.choose_fixed_endpoint()
        c = self.fixed_endpoint
        x_prev = self.variable_endpoint
        iteration = 0
        while True:
            denominator = self.f(x_prev) - self.f(c)
            if denominator == 0:
                raise ZeroDivisionError("Деление на ноль в методе хорд.")
            x_next = x_prev - (x_prev - c) * self.f(x_prev) / denominator
            iteration += 1
            if abs(x_next - x_prev) < self.eps:
                break
            x_prev = x_next
            if iteration > 10000:
                raise RuntimeError("Превышено максимальное число итераций.")
        return x_next, self.f(x_next), iteration


class NewtonMethodSolver:
    def __init__(self, f, df, f2, a: float, b: float, eps: float):
        """
        Инициализация решателя методом Ньютона.
        Параметры:
          f - функция
          df - её производная
          f2 - вторая производная функции f (используется для выбора начального приближения)
          a, b - границы интервала (при a > b они меняются местами)
          eps - требуемая точность
        """
        self.f = f
        self.df = df
        self.f2 = f2
        self.eps = eps
        self.a, self.b = (a, b) if a <= b else (b, a)

    def count_sign_changes(self, num_points=1000) -> int:
        xs = np.linspace(self.a, self.b, num_points)
        f_vals = self.f(xs)
        signs = np.sign(f_vals)
        changes = 0
        for i in range(1, len(signs)):
            if signs[i] == 0 or signs[i - 1] == 0:
                continue
            if signs[i] != signs[i - 1]:
                changes += 1
        return changes

    def check_root_existence(self) -> None:
        sign_changes = self.count_sign_changes()
        if sign_changes != 1:
            raise ValueError(
                f"На интервале [{self.a}, {self.b}] обнаружено {sign_changes} смен(ы) знака, что не гарантирует наличие ровно одного корня."
            )

    def choose_initial_approximation(self) -> float:
        """
        Выбирает начальное приближение для метода Ньютона.
        Рекомендуется выбрать ту границу, для которой выполнено условие f(x)*f''(x) > 0.
        """
        if self.f(self.a) * self.f2(self.a) > 0:
            return self.a
        elif self.f(self.b) * self.f2(self.b) > 0:
            return self.b
        else:
            raise ValueError(
                "Невозможно выбрать начальное приближение: условие f(x)*f''(x) > 0 не выполнено ни для одной из границ.")

    def solve(self) -> (float, float, int):
        """
        Выполняет итерационный процесс метода Ньютона:
          x_(n+1) = x_n - f(x_n)/f'(x_n)
        Возвращает кортеж: (найденный корень, значение функции в корне, число итераций).
        """
        self.check_root_existence()
        x_prev = self.choose_initial_approximation()
        iteration = 0
        while True:
            derivative = self.df(x_prev)
            if derivative == 0:
                raise ZeroDivisionError("Производная равна нулю при x = " + str(x_prev))
            x_next = x_prev - self.f(x_prev) / derivative
            iteration += 1
            if abs(x_next - x_prev) < self.eps:
                break
            x_prev = x_next
            if iteration > 10000:
                raise RuntimeError("Превышено максимальное число итераций.")
        return x_next, self.f(x_next), iteration


# ---------------------------
# Функция для построения графика
# ---------------------------
def plot_function(f, a: float, b: float, root: float = None):
    """
    Строит график функции f на интервале [a, b].
    Если root задан, отмечает его на графике.
    """
    xs = np.linspace(a, b, 1000)
    ys = f(xs)

    plt.figure()
    plt.plot(xs, ys, label="f(x)")
    if root is not None:
        plt.scatter([root], [f(root)], color='red', zorder=5, label="Найденный корень")
    plt.xlim(a, b)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("График функции на интервале [{}, {}]".format(a, b))
    plt.grid(True)
    plt.legend()
    plt.show()


# ---------------------------
# Основная функция
# ---------------------------
def main():
    # Выбор функции
    print("Выберите функцию для поиска корня:")
    for key, func in functions.items():
        print(f"{key} - {func['name']}")
    func_choice = input("Введите номер функции (1-4): ").strip()
    if func_choice not in functions:
        print("Неверный выбор функции.")
        return
    selected = functions[func_choice]
    f_selected = selected["f"]
    df_selected = selected["df"]
    f2_selected = selected["f2"]

    # Ввод интервала и погрешности
    try:
        a = float(input("Введите левую границу интервала a: "))
        b = float(input("Введите правую границу интервала b: "))
        eps = float(input("Введите требуемую погрешность epsilon: "))
    except ValueError:
        print("Некорректный ввод.")
        return

    # Выбор метода
    print("Выберите метод для поиска корня:")
    print("1 - Метод простой итерации")
    print("2 - Метод хорд")
    print("3 - Метод Ньютона")
    method_choice = input("Введите номер метода (1, 2 или 3): ").strip()

    try:
        if method_choice == "1":
            solver = FixedPointIterationSolver(f_selected, df_selected, a, b, eps)
        elif method_choice == "2":
            solver = ChordMethodSolver(f_selected, f2_selected, a, b, eps)
        elif method_choice == "3":
            solver = NewtonMethodSolver(f_selected, df_selected, f2_selected, a, b, eps)
        else:
            print("Неверный выбор метода.")
            return

        root, f_at_root, iterations = solver.solve()
    except (ValueError, RuntimeError, ZeroDivisionError) as e:
        print(f"Ошибка: {e}")
        return

    print("\nРезультаты вычислений:")
    print(f"Выбранная функция: {selected['name']}")
    print(f"Найденный корень: {root}")
    print(f"Значение функции в корне: {f_at_root}")
    print(f"Количество итераций: {iterations}")

    # Построение графика выбранной функции на заданном интервале
    plot_function(f_selected, a, b, root)


if __name__ == "__main__":
    main()

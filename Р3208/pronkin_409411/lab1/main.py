import math
import sys
import random
from itertools import permutations


def print_augmented_matrix(A, b):
    """
    Функция для вывода расширенной матрицы [A|b].
    """
    n = len(A)
    print("Расширенная матрица [A|b]:")
    for i in range(n):
        row_str = ""
        for j in range(n):
            row_str += f"{A[i][j]:10.4f} "
        row_str += f"| {b[i]:10.4f}"
        print(row_str)
    print()


def euclidian_norm(v):
    """
    Функция вычисления евклидовой нормы вектора.
    """
    return math.sqrt(sum(val * val for val in v))


def has_diagonal_dominance(A):
    """
    Проверяет, есть ли у матрицы A диагональное преобладание.
    Возвращает True, если преобладание есть, иначе False.
    """
    n = len(A)
    for i in range(n):
        diagonal_element = abs(A[i][i])
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diagonal_element < row_sum:
            return False
    return True


def swap_rows(A, b, row1, row2):
    """
    Меняет местами две строки row1 и row2 в матрице A и векторе b.
    """
    A[row1], A[row2] = A[row2], A[row1]
    b[row1], b[row2] = b[row2], b[row1]


def make_diagonally_dominant(A, b):
    """
    Пытается привести матрицу A к диагонально преобладающей перестановкой строк.
    Если это невозможно, выводит ошибку.
    """
    n = len(A)
    for perm in permutations(range(n)):
        A_permuted = [A[i] for i in perm]
        b_permuted = [b[i] for i in perm]

        if has_diagonal_dominance(A_permuted):
            for i in range(n):
                A[i] = A_permuted[i]
                b[i] = b_permuted[i]
            return True
    return False


def build_Bc(A, b):
    """
    Функция формирования матрицы B и вектора c
    для метода простых итераций (x = Bx + c).
    Предполагается, что на главной диагонали нет нулевых элементов.
    """
    n = len(A)
    B = [[0.0 for _ in range(n)] for _ in range(n)]
    c = [0.0 for _ in range(n)]

    for i in range(n):
        diag = A[i][i]
        c[i] = b[i] / diag
        for j in range(n):
            if j != i:
                B[i][j] = -A[i][j] / diag
    return B, c


def simple_iteration_method(B, c, eps, max_iter):  # можно sys.maxsize на max_iter повесить
    """
    Основная функция итерационного процесса.
    Возвращает кортеж (x, num_iterations), где:
    x - найденный вектор решения,
    num_iterations - реальное число итераций.
    """
    n = len(B)
    x = [0.0 for _ in range(n)]
    x_new = [0.0 for _ in range(n)]

    # x = Bx + c

    #x^(k+1) = x^(k)

    for k in range(max_iter):
        for i in range(n):
            temp_sum = 0.0
            for j in range(n):
                temp_sum += B[i][j] * x[j]
            x_new[i] = temp_sum + c[i]

        # Вычисляем вектор погрешностей: |x^(k) - x^(k-1)|
        error_vector = [abs(x_new[i] - x[i]) for i in range(n)]

        # Проверяем критерий остановки: если все абсолютные ошибки < eps
        if all(error < eps for error in error_vector):
            return x_new, k, error_vector  # Возвращаем найденное решение и количество итераций

        # x = x_new.copy()
        x = x_new[:]

    return x, max_iter, error_vector

def read_matrix_from_stdout():
    """
    Функция для чтения матрицы A и вектора b из файла.
        Формат файла:
        n
        A[0][0] A[0][1] ... A[0][n-1] b[0]
        A[1][0] A[1][1] ... A[1][n-1] b[1]
        ...
        A[n-1][0] A[n-1][1] ... A[n-1][n-1] b[n-1]
        eps (точность вычислений)
        max_iter (необязательный параметр, если отсутствует, ставим "бесконечность")
    Читает
    :return: кортеж из
                    A – матрицы
                    b - вектора
                    eps - точности
                    max_iter - максимального кол-ва итераций
    """
    # размер матрицы n
    while True:
        try:
            n = int(input("Введите размер матрицы (от 1 до 20): "))
            if 1 <= n <= 20:
                break
            else:
                print("Число вне диапазона [1..20]. Повторите ввод.")
        except ValueError:
            print("Некорректный ввод. Попробуйте ещё раз.")

    # матрица A и вектор b
    print("Введите элементы матрицы A и вектора b (A|b):")
    A = []
    b = []
    for i in range(n):
        while True:
            try:
                row = list(map(float, input().split()))
                if len(row) != n + 1:
                    raise ValueError
                A.append(row[:-1])  # Первые n чисел — матрица A
                b.append(row[-1])  # Последнее число — вектор b
                break
            except ValueError:
                print("Некорректный ввод. Введите ровно", n + 1, "чисел.")

    # точность
    while True:
        try:
            eps = float(input("Введите желаемую точность: "))
            break
        except ValueError:
            print("Некорректный ввод. Введите число.")

    # максимальное число итераций
    while True:
        try:
            line = input("Введите максимальное число итераций(нажмите ENTER для неограниченных итераций): ")
            if line == "":
                max_iter = sys.maxsize
                break
            max_iter = int(line)
            if max_iter <= 0:
                raise ValueError
            break
        except ValueError:
            print("Некорректный ввод. Введите натуральное число.")
    return A, b, eps, max_iter


def read_matrix_from_file(filename):
    """
    Функция для чтения матрицы A и вектора b из файла.
    Формат файла:
    n
    A[0][0] A[0][1] ... A[0][n-1] b[0]
    A[1][0] A[1][1] ... A[1][n-1] b[1]
    ...
    A[n-1][0] A[n-1][1] ... A[n-1][n-1] b[n-1]
    eps (точность вычислений)
    max_iter (необязательный параметр, если отсутствует, ставим "бесконечность")
    :param filename: имя файла, который читаем
    :return: кортеж из
                A – матрицы
                b - вектора
                eps - точности
                max_iter - максимального кол-ва итераций
    """
    with open(filename, "r") as file:
        lines = file.readlines()

    # Считываем размер матрицы n
    n = int(lines[0].strip())
    print(f"Размер данной матрицы: {n}")

    A = []
    b = []

    # матрица A и вектор b
    for i in range(1, n + 1):
        values = list(map(float, lines[i].split()))

        if len(values) != n + 1:
            raise ValueError(f"Ошибка: строка {i} должна содержать {n + 1} чисел, а не {len(values)}.")

        A.append(values[:-1])  # Первые n чисел — это строка матрицы A
        b.append(values[-1])  # Последнее число — элемент вектора b

    # точность
    eps = float(lines[n + 1].strip())
    # кол-во итераций
    if len(lines) > n + 2:
        max_iter = int(lines[n + 2].strip())
    else:
        max_iter = sys.maxsize
    return A, b, eps, max_iter


def generate_random_matrix(n, min_value=-10, max_value=10):
    """
    Генерирует случайную матрицу A размером n x n с диагональным преобладанием
    и случайный вектор b длиной n.
    min_value и max_value — диапазон случайных чисел.
    :param n:
    :param min_value:
    :param max_value:
    :return:
    """
    A = [[random.uniform(min_value, max_value) for _ in range(n)] for _ in range(n)]
    b = [random.uniform(min_value, max_value) for _ in range(n)]

    # диагональное преобладание
    for i in range(n):
        A[i][i] = sum(abs(A[i][j]) for j in range(n) if i != j) + random.uniform(1, 5)  # Диагональ > суммы остальных

    return A, b


def main():
    # Пользователь выбирает способ ввода данных
    choice = 0
    while choice not in [1, 2, 3]:
        try:
            choice = int(input("Выберите способ ввода данных:\n"
                               "1 - Ввести с клавиатуры\n"
                               "2 - Загрузить из файла\n"
                               "3 - Сгенерировать случайную матрицу\n"
                               "Введите номер варианта: ").strip())
        except ValueError:
            print("Ошибка! Попробуйте ещё раз.")

    if choice == 1:
        A, b, eps, max_iter = read_matrix_from_stdout()
    elif choice == 2:
        filename = input("Введите имя файла (например, matrix.txt): ").strip()
        try:
            A, b, eps, max_iter = read_matrix_from_file(filename)
        except Exception as e:
            print(f"Ошибка чтения файла: {e}")
            return
    else:
        while True:
            try:
                n = int(input("Введите размер случайной матрицы (от 1 до 20): "))
                if 1 <= n <= 20:
                    break
                else:
                    print("Число вне диапазона [1..20]. Повторите ввод.")
            except ValueError:
                print("Ошибка! Введите число.")

        A, b = generate_random_matrix(n)

        # Ввод точности и числа итераций
        while True:
            try:
                eps = float(input("Введите желаемую точность: "))
                break
            except ValueError:
                print("Ошибка! Введите число.")

        while True:
            try:
                line = input("Введите максимальное число итераций (ENTER для бесконечности): ")
                if line == "":
                    max_iter = sys.maxsize
                    break
                max_iter = int(line)
                if max_iter <= 0:
                    raise ValueError
                break
            except ValueError:
                print("Ошибка! Введите натуральное число.")

    # выведем матрицу
    print_augmented_matrix(A, b)

    # диагональное преобразование
    if not has_diagonal_dominance(A):
        print("Матрица не обладает диагональным преобладанием. Пробуем перестановку строк...")
        if not make_diagonally_dominant(A, b):
            print("Не удалось достигнуть диагонального преобладания. Завершаем работу программы...")
            return
        else:
            print("Матрица с диагональным преобладанием:")
            print_augmented_matrix(A, b)

    # Строим матрицу B и вектор c
    B, c_vec = build_Bc(A, b)

    # метод простых итераций
    x_solution, real_iter, error_vector = simple_iteration_method(B, c_vec, eps, max_iter)

    # результат
    print("\nРезультат (метод простых итераций):")
    for i, val in enumerate(x_solution):
        print(f"x[{i}] = {val:12.6f}")
    # print(f"Норма матрицы: {}")
    print(f"Число итераций: {real_iter}")
    print("Вектор погрешности:", " ".join(f"{e:10.6f}" for e in error_vector))


if __name__ == "__main__":
    main()

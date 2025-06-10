# ---------------------------------------
# Функции, интегралы которых вычисляем
# ---------------------------------------
def f(x):
    """
    3(x)^2
    """
    return 3 * x ** 2


def f2(x):
    """
    3(x)^3+4x
    """
    return 3 * x ** 3 + 4 * x


def f3(x):
    """
    3(x)^2+4x+5 
    """
    return 3 * x ** 2 + 4 * x + 5


# ---------------------------------------
# Методы вычисления интегралов
# ---------------------------------------

# ---------------------------------------
# прямоугольники
# ---------------------------------------
def left_rectangle_method(func, a, b, n):
    """
    Вычисление интеграла методом левых прямоугольников.

    Параметры:
      func: интегрируемая функция,
      a, b: пределы интегрирования,
      n: число разбиений.

    Возвращает:
      Приближённое значение интеграла.
    """
    h = (b - a) / n
    total = 0.0
    for i in range(n):
        x_i = a + i * h  # левая точка подинтервала
        total += func(x_i)
    return total * h


def right_rectangle_method(func, a, b, n):
    """
    Вычисление интеграла методом правых прямоугольников.

    Параметры:
      func: интегрируемая функция,
      a, b: пределы интегрирования,
      n: число разбиений.

    Возвращает:
      Приближённое значение интеграла.
    """
    h = (b - a) / n
    total = 0.0
    for i in range(1, n + 1):
        x_i = a + i * h  # правая точка подинтервала
        total += func(x_i)
    return total * h


def middle_rectangle_method(func, a, b, n):
    """
    Вычисление интеграла методом центральных прямоугольников.

    Параметры:
      func: интегрируемая функция,
      a, b: пределы интегрирования,
      n: число разбиений.

    Возвращает:
      Приближённое значение интеграла.
    """
    h = (b - a) / n
    total = 0.0
    for i in range(n):
        x_i = a + i * h + h / 2  # центральная точка подинтервала
        total += func(x_i)
    return total * h


# ---------------------------------------
# трапеция
# ---------------------------------------
def trapezoid_method(func, a, b, n):
    """
    Вычисление интеграла методом трапеций.

    Параметры:
      func: интегрируемая функция,
      a, b: пределы интегрирования,
      n: число разбиений.

    Возвращает:
      Приближённое значение интеграла.
    """
    h = (b - a) / n
    total = (func(a) + func(b)) / 2.0
    for i in range(1, n):
        x_i = a + i * h  # левая точка подинтервала
        total += func(x_i)
    return total * h


# ---------------------------------------
# симпсон
# ---------------------------------------
def simpson_method(func, a, b, n):
    """
    Вычисление интеграла методом Симпсона.

    Параметры:
      func: интегрируемая функция,
      a, b: пределы интегрирования,
      n: число разбиений.

    Возвращает:
      Приближённое значение интеграла.
    """
    h = (b - a) / n
    total = func(a) + func(b)
    for i in range(1, n, 2):
        x_i = a + i * h  # левая точка подинтервала
        total += 4 * func(x_i)
    for i in range(2, n, 2):
        x_i = a + i * h
        total += 2 * func(x_i)
    return total * h / 3


# ---------------------------------------
# Вычисление интеграла с правилом Рунге
# ---------------------------------------
def integrate_with_runge(method, func, a, b, eps, order=1, initial_n=4):
    """
    Вычисление интеграла с использованием заданного метода и правила Рунге для 
    оценки погрешности.

    Параметры:
      method: функция для вычисления интеграла (например, left_rectangle_method или
       right_rectangle_method),
      func: интегрируемая функция,
      a, b: пределы интегрирования,
      eps: требуемая точность,
      order: порядок метода (для прямоугольников order = 1),
      initial_n: начальное число разбиений (по умолчанию 4).

    Возвращает:
      Кортеж (интегральное приближение, итоговое число разбиений).
    """
    n = initial_n
    I_n = method(func, a, b, n)
    n *= 2
    I_2n = method(func, a, b, n)

    # Правило Рунге: |I(2n) - I(n)| / (2^order - 1) < eps
    while abs(I_2n - I_n) / (2 ** order - 1) > eps:
        I_n = I_2n
        n *= 2
        I_2n = method(func, a, b, n)

    return I_2n, n


def main():
    valid_choices = {'1', '2', '3'}

    function_choice = None
    function_to_compute = None
    print("Вычисление интеграла функции")
    print("Выберите функцию, которую хотите проинтегрировать:")
    print("1 - 3x^2")
    print("2 - 3x^3 + 4x")
    print("3 - 3x^2 + 4x + 5")

    while function_choice not in valid_choices:
        function_choice = input("Введите номер функции: ").strip()
        if function_choice not in valid_choices:
            print("Неверный ввод. Пожалуйста, введите 1, 2 или 3.")

    if function_choice == '1':
        function_to_compute = f
    elif function_choice == '2':
        function_to_compute = f2
    elif function_choice == '3':
        function_to_compute = f3


    print("Выберите метод вычисления интеграла:")
    print("1 - Метод прямоугольников")
    print("2 - Метод трапеций")
    print("3 - Метод Симпсона")

    choice = None

    while choice not in valid_choices:
        choice = input("Введите номер метода: ").strip()
        if choice not in valid_choices:
            print("Неверный ввод. Пожалуйста, введите 1, 2 или 3.")

    if choice == "1":
        print("Какой прямоугольник будем использовать для вычислений?")
        print("1 - Левый")
        print("2 - Средний")
        print("3 - Правый")

        choice2 = None

        while choice2 not in valid_choices:
            choice2 = input("Введите номер метода: ").strip()
            if choice2 not in valid_choices:
                print("Неверный ввод. Пожалуйста, введите 1, 2 или 3.")

        if choice2 == "1":
            integration_method = left_rectangle_method
            order = 1  # порядок метода для прямоугольников (левый)
        elif choice2 == "2":
            integration_method = middle_rectangle_method
            order = 2  # порядок метода для прямоугольников (средний)
        elif choice2 == "3":
            integration_method = right_rectangle_method
            order = 1  # порядок метода для прямоугольников (правый)
    elif choice == "2":
        integration_method = trapezoid_method
        order = 2
    elif choice == "3":
        integration_method = simpson_method
        order = 4
    else:
        print("Некорректный выбор метода.")
        return

    try:
        a = float(input("Введите нижний предел интегрирования (a): "))
        b = float(input("Введите верхний предел интегрирования (b): "))
        eps = float(input("Введите требуемую точность (eps): "))
    except ValueError:
        print("Ошибка: необходимо вводить числовые значения.")
        return

    result, subdivisions = integrate_with_runge(
        integration_method, function_to_compute, a, b, eps, initial_n=4, order=order
    )

    print("\nРезультаты вычисления:")
    print(f"Приближённое значение интеграла: {result}")
    print(f"Число разбиений для достижения требуемой точности: {subdivisions}")


if __name__ == "__main__":
    main()

def main():
    print("Выберите, что решать:")
    print("1 - Одно нелинейное уравнение")
    print("2 - Система нелинейных уравнений")
    choice = input("Введите номер задачи (1 или 2): ").strip()

    if choice == "1":
        try:
            import single_equations
            single_equations.main()
        except ImportError:
            print("Ошибка импорта модуля для решения уравнений.")
    elif choice == "2":
        try:
            import systems
            systems.main()
        except ImportError:
            print("Ошибка импорта модуля для решения систем уравнений.")
    else:
        print("Неверный выбор. Запустите программу снова и введите 1 или 2.")


if __name__ == "__main__":
    main()

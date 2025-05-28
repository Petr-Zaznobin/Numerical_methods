import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Класс Phi оставляем без изменений (для тестирования, вариант с кусочно‑определённой функцией)
class Phi:
    def func(self, x):
        if -1 <= x < 0:
            return x**3 + 3 * x**2
        elif 0 <= x <= 1:
            return -x**3 + 3 * x**2
        # Если за пределами [-1,1], возвращаем 0 (или можно вызвать ошибку)
        #return 0.0

    def derivative(self, x):
        if -1 <= x < 0:
            return 3 * x**2 + 6 * x
        elif 0 <= x <= 1:
            return -3 * x**2 + 6 * x
        return 0.0

    def derivative_2(self, x):
        if -1 <= x < 0:
            return 6 * x + 6
        elif 0 <= x <= 1:
            return -6 * x + 6
        return 0.0

# Класс Main_func для варианта 14: f(x) = sqrt(1 + x^4)
class Main_func:
    def func(self, x):
        # f(x) = sqrt(1 + x^4)
        return np.sqrt(1.0 + x**4)

    def derivative(self, x):
        # f'(x) = (1/2)(1 + x^4)^(-1/2) * 4x^3 = (2x^3) / sqrt(1 + x^4)
        return (2.0 * x**3) / np.sqrt(1.0 + x**4)

    def derivative_2(self, x):
        # f''(x) = 6x^2 / sqrt(1 + x^4) - 4x^6 / (1 + x^4)^(3/2)
        # упрощённая форма: (2x^2 (3 + x^4)) / (1 + x^4)^(3/2)
        numerator = 2.0 * x**2 * (3.0 + x**4)
        denominator = (1.0 + x**4)**(3.0/2.0)
        return numerator / denominator

# Класс Main_Func добавляет колебательный член cos(10x)
class Main_Func2:
    def func(self, x):
        return Main_func().func(x) + np.cos(10 * x)

    def derivative(self, x):
        return Main_func().derivative(x) - 10 * np.sin(10 * x)

    def derivative_2(self, x):
        return Main_func().derivative_2(x) - 100 * np.cos(10 * x)

# Класс для построения кубического сплайна и сравнения с исходной функцией
class Spline:
    def __init__(self, n, a, b, eps, func):
        self.n = n
        self.a = a
        self.b = b
        self.h = (b - a) / n
        self.av = np.zeros(n + 1)
        self.bv = np.zeros(n + 1)
        self.cv = np.zeros(n + 1)
        self.dv = np.zeros(n + 1)
        self.last_eps = np.inf
        self.eps = eps
        self.func = func
        self.max_error_spline = []
        self.max_error_derivative = []
        self.max_error_derivative_2 = []

        # Вычисляем значения функции в узлах сетки
        for i in range(n + 1):
            self.av[i] = func.func(a + self.h * i)


    def count_coeffs(self):
        alpha = np.zeros(self.n)
        beta = np.zeros(self.n)

        # Прямой ход метода трапеций для трёхдиагональной системы
        for i in range(1, self.n):
            denom = (self.h * alpha[i - 1] + 4 * self.h)
            alpha[i] = -self.h / denom
            x = self.a + self.h * i
            x_prev = x - self.h
            x_next = x + self.h
            Fi = 6 * (self.func.func(x_next) - 2 * self.func.func(x) + self.func.func(x_prev)) / self.h
            beta[i] = (Fi - self.h * beta[i - 1]) / denom

        # Обратный ход для поиска c_i
        for i in reversed(range(1, self.n)):
            self.cv[i] = alpha[i] * self.cv[i + 1] + beta[i]

        # Вычисляем d_i и b_i
        for i in range(1, self.n + 1):
            self.dv[i] = (self.cv[i] - self.cv[i - 1]) / self.h
            self.bv[i] = (
                (self.av[i] - self.av[i - 1]) / self.h
                + self.cv[i] * self.h / 3.0
                + self.cv[i - 1] * self.h / 6.0
            )

        # Формируем DataFrame коэффициентов и возвращаем
        grid = [self.a + i * self.h for i in range(self.n + 1)]
        coeffs = {
            'i':      list(range(1, self.n + 1)),
            'x_{i-1}': grid[0:-1],
            'x_i':     grid[1:],
            'a':       self.av[1:],
            'b':       self.bv[1:],
            'c':       self.cv[1:],
            'd':       self.dv[1:]
        }
        self.df_coeffs = pd.DataFrame(coeffs)
        return self.df_coeffs

    def plot_spline(self, ticks):
        for i in range(1, self.n + 1):
            x_left = self.a + self.h * (i - 1)
            x_right = x_left + self.h
            x_space = np.linspace(x_left, x_right, ticks)

            # Сплайн на отрезке [x_{i-1}, x_i]
            spline_seg = [
                self.av[i]
                + self.bv[i] * (x - x_right)
                + self.cv[i] / 2.0 * (x - x_right)**2
                + self.dv[i] / 6.0 * (x - x_right)**3
                for x in x_space
            ]
            plt.plot(x_space, spline_seg, color="purple", linewidth=2.0)

            # Исходная функция
            func_seg = [self.func.func(x) for x in x_space]
            plt.plot(x_space, func_seg, color="yellow", linewidth=1.5)

        xticks = [self.a + i * self.h for i in range(self.n + 1)]
        plt.xticks(xticks if self.n < 10 else None)
        plt.grid()
        plt.title("F(x) vs S(x)")
        #plt.show()

    def plot_derivative(self, ticks):
        for i in range(1, self.n + 1):
            x_left = self.a + self.h * (i - 1)
            x_right = x_left + self.h
            x_space = np.linspace(x_left, x_right, ticks)

            spline_deriv = [
                self.bv[i]
                + self.cv[i] * (x - x_right)
                + self.dv[i] / 2.0 * (x - x_right)**2
                for x in x_space
            ]
            plt.plot(x_space, spline_deriv, color="purple", linewidth=2.0)

            func_deriv = [self.func.derivative(x) for x in x_space]
            plt.plot(x_space, func_deriv, color="yellow", linewidth=1.5)

        xticks = [self.a + i * self.h for i in range(self.n + 1)]
        plt.xticks(xticks if self.n < 10 else None)
        plt.grid()
        plt.title("F'(x) vs S'(x)")
        #plt.show()

    def plot_derivative_2(self, ticks):
        for i in range(1, self.n + 1):
            x_left = self.a + self.h * (i - 1)
            x_right = x_left + self.h
            x_space = np.linspace(x_left, x_right, ticks)

            spline_deriv2 = [
                self.cv[i] + self.dv[i] * (x - x_right) for x in x_space
            ]
            plt.plot(x_space, spline_deriv2, color="purple", linewidth=2.0)

            func_deriv2 = [self.func.derivative_2(x) for x in x_space]
            plt.plot(x_space, func_deriv2, color="yellow", linewidth=1.5)

        xticks = [self.a + i * self.h for i in range(self.n + 1)]
        plt.xticks(xticks if self.n < 10 else None)
        plt.grid()
        plt.title("F''(x) vs S''(x)")
        #plt.show()

    def coeffs_to_table(self):
        grid = [self.a + i * self.h for i in range(self.n + 1)]
        coeffs = {
            'Xi-1': grid[0:-1],
            'Xi':   grid[1:],
            'a':    self.av[1:],
            'b':    self.bv[1:],
            'c':    self.cv[1:],
            'd':    self.dv[1:]
        }
        self.df_coeffs = pd.DataFrame(coeffs)
        #print(self.df_coeffs)
        #print(coeffs)
        return self.df_coeffs

    def collect_errors_table(self, ticks):
        data = []
        for i in range(1, self.n + 1):
            x_left = self.a + (i - 1) * self.h
            x_right = x_left + self.h
            x_space = np.linspace(x_left, x_right, ticks)

            spline_seg = (
                self.av[i]
                + self.bv[i] * (x_space - x_right)
                + self.cv[i] / 2.0 * (x_space - x_right)**2
                + self.dv[i] / 6.0 * (x_space - x_right)**3
            )
            func_seg = np.array([self.func.func(x) for x in x_space])
            max_err_func = np.max(np.abs(spline_seg - func_seg))

            spline_deriv = (
                self.bv[i]
                + self.cv[i] * (x_space - x_right)
                + self.dv[i] / 2.0 * (x_space - x_right)**2
            )
            func_deriv = np.array([self.func.derivative(x) for x in x_space])
            max_err_deriv = np.max(np.abs(spline_deriv - func_deriv))

            spline_deriv2 = self.cv[i] + self.dv[i] * (x_space - x_right)
            func_deriv2 = np.array([self.func.derivative_2(x) for x in x_space])
            max_err_deriv2 = np.max(np.abs(spline_deriv2 - func_deriv2))

            data.append({
                'i': i,
                 #'x_{i-1}': x_left,
                'x_i':     x_right,
                'max|F(x)-S(x)|':    max_err_func,
                "max|F'(x)-S'(x)|":  max_err_deriv,
                "max|F''(x)-S''(x)|": max_err_deriv2
            })

        self.df_errors = pd.DataFrame(data)
        return self.df_errors

    def plot_error_spline(self, ticks):
        self.max_error_spline = []
        for i in range(1, self.n + 1):
            x_left = self.a + self.h * (i - 1)
            x_right = x_left + self.h
            x_space = np.linspace(x_left, x_right, ticks)

            spline_seg = [
                self.av[i]
                + self.bv[i] * (x - x_right)
                + self.cv[i] / 2.0 * (x - x_right)**2
                + self.dv[i] / 6.0 * (x - x_right)**3
                for x in x_space
            ]
            func_seg = [self.func.func(x) for x in x_space]
            error_seg = np.abs(np.array(spline_seg) - np.array(func_seg))
            plt.plot(x_space, error_seg, color="purple", linewidth=1.5)
            self.max_error_spline.append(np.max(error_seg))

        xticks = [self.a + i * self.h for i in range(self.n + 1)]
        plt.xticks(xticks if self.n < 10 else None)
        plt.grid()
        plt.title("Погрешность |F(x) - S(x)|")
        #plt.show()

    def plot_error_derivative(self, ticks):
        self.max_error_derivative = []
        for i in range(1, self.n + 1):
            x_left = self.a + self.h * (i - 1)
            x_right = x_left + self.h
            x_space = np.linspace(x_left, x_right, ticks)

            spline_deriv = [
                self.bv[i]
                + self.cv[i] * (x - x_right)
                + self.dv[i] / 2.0 * (x - x_right)**2
                for x in x_space
            ]
            func_deriv = [self.func.derivative(x) for x in x_space]
            error_deriv = np.abs(np.array(spline_deriv) - np.array(func_deriv))
            plt.plot(x_space, error_deriv, color="purple", linewidth=1.5)
            self.max_error_derivative.append(np.max(error_deriv))

        xticks = [self.a + i * self.h for i in range(self.n + 1)]
        plt.xticks(xticks if self.n < 10 else None)
        plt.grid()
        plt.title("Погрешность |F'(x) - S'(x)|")
        #plt.show()

    def plot_error_derivative_2(self, ticks):
        self.max_error_derivative_2 = []
        for i in range(1, self.n + 1):
            x_left = self.a + self.h * (i - 1)
            x_right = x_left + self.h
            x_space = np.linspace(x_left, x_right, ticks)

            spline_deriv2 = [
                self.cv[i] + self.dv[i] * (x - x_right) for x in x_space
            ]
            func_deriv2 = [self.func.derivative_2(x) for x in x_space]
            error_deriv2 = np.abs(np.array(spline_deriv2) - np.array(func_deriv2))
            plt.plot(x_space, error_deriv2, color="purple", linewidth=1.5)
            self.max_error_derivative_2.append(np.max(error_deriv2))

        xticks = [self.a + i * self.h for i in range(self.n + 1)]
        plt.xticks(xticks if self.n < 10 else None)
        plt.grid()
        plt.title("Погрешность |F''(x) - S''(x)|")
        #plt.show()

    def print_results(self, ticks):
        self.plot_spline(ticks)
        self.plot_derivative(ticks)
        self.plot_derivative_2(ticks)
        self.plot_error_spline(ticks)
        self.plot_error_derivative(ticks)
        self.plot_error_derivative_2(ticks)

        self.coeffs_to_table()
        self.collect_errors_table(ticks)
        np.set_printoptions(formatter={'float_kind': '{:e}'.format})
        print(f"Сетка сплайна: n = {self.n}")
        print(f"Контрольная сетка: n = {self.n * 4}")
        print(f"max|F(x) - S(x)| = {np.array(self.max_error_spline).max():e}")
        print(f"max|F'(x) - S'(x)| = {np.array(self.max_error_derivative).max():e}")
        print(f"max|F''(x) - S''(x)| = {np.array(self.max_error_derivative_2).max():e}")

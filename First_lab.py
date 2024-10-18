import numpy as np
import matplotlib.pyplot as plt

STEP = 0.0001  # Константа шага
P = 4  # порядок метода
EPS = 0.01  # епсилон граничное


class one_d_diffur:
    def __init__(self, category: int, variant_num: int, u0: float, x0: float, n_max: int, eps: float, b: float):
        """
        int category   : тип ДУ задания (0-тестовое, 1-первое)
        int variant_num: номер варианта,
        double u0      : начальное условие
        double x0      : начальное значение x
        int n_max      : количество итераций
        float eps      : граничное значение
        float b        : ограничение вычислений по ОХ
        """

        # инициализация полей
        self.variant_num = variant_num
        self.u0 = u0
        self.x0 = x0
        self.u = u0
        self.x = x0
        self.n_max = n_max
        self.eps = eps
        self.h = STEP
        self.category = category
        self.b = b
        if self.category == 0:
            self.data = np.empty((0, 11))
            # [index, xi, vi, v2i, vi-v2i, ОЛП, hi, C1, C2, ui, |ui-vi|]
        else:
            self.data = np.empty((0, 9))
            # [index, xi, vi, vi-v2i, ОЛП, hi, C1, C2]
        self.c1 = 0
        self.c2 = 0

    # функция вывода таблицы
    def print_table(self):
        print(self.data)

    # Функции (тестовая, основная задача)
    def __f(self, x, u):
        category = self.category
        if category == 0:
            dudx = (-1) ** self.variant_num * (0.5 * u * self.variant_num)  # -0.5*u
            return dudx
        elif category == 1:
            dudx = u ** 2 / (1 + x ** 2) + u - u ** 3 * np.sin(10 * x)
            return dudx

    # вычисление без локальной погрешности
    def __solve_without_lp(self):
        v = self.u
        x = self.x
        k1 = self.__f(x, v)  # -0.5*v
        k2 = self.__f(x + 0.5 * self.h, v + 0.5 * self.h * k1)  # -0.5*(v-0.25*v*STEP)
        k3 = self.__f(x + 0.5 * self.h, v + 0.5 * self.h * k2)
        k4 = self.__f(x + self.h, v + self.h * k3)
        v = v + 1 / 6 * self.h * (k1 + 2 * k2 + 2 * k3 + k4)
        x = x + self.h
        return x, v

    # вспомогательная функция нахождения ki
    def __search_k_for_solve_with_lp(self, x, v):
        k1 = self.__f(x, v)
        k2 = self.__f(x + 0.25 * self.h, v + 0.25 * self.h * k1)
        k3 = self.__f(x + 0.25 * self.h, v + 0.25 * self.h * k2)
        k4 = self.__f(x + 0.5 * self.h, v + 0.5 * self.h * k3)
        return k1, k2, k3, k4

    # вычисления методом с локальной погрешностью
    def __solve_with_lp(self):
        v = self.u
        x = self.x
        # первая часть для 1/2 STEP
        k1, k2, k3, k4 = self.__search_k_for_solve_with_lp(x, v)
        v1 = v + 1 / 12 * self.h * (k1 + 2 * k2 + 2 * k3 + k4)
        x = x + 0.5 * self.h
        # вторая часть для еще одного 1/2 STEP
        k1, k2, k3, k4 = self.__search_k_for_solve_with_lp(x, v)
        v = v1 + 1 / 12 * self.h * (k1 + 2 * k2 + 2 * k3 + k4)
        x = x + 0.5 * self.h
        return x, v

    # основная функция решения ДУ
    def solve(self, with_lp: bool):
        """
        bool with_lp: флан на решение с локальной погрешностью
        """

        # заполнение строк таблиц
        if self.category == 0:
            self.data = np.vstack([self.data, [0, self.x, self.u, self.u, 0, 0, self.h, self.c1, self.c2, 0, 0]])
        else:
            self.data = np.vstack([self.data, [0, self.x, self.u, self.u, 0, 0, self.h, self.c1, self.c2]])

        # Работа с погрешностью
        prev_v = self.u
        if with_lp:
            for iteration in range(0, self.n_max):
                x1, v1 = self.__solve_with_lp()
                x2, v2 = self.__solve_without_lp()
                s = (v1 - v2) / (2 ** P - 1)
                s = abs(s)
                if self.eps >= s >= self.eps / (2 ** (P + 1)):
                    if self.category == 0:
                        ui = self.true_solve_test(x1)
                        self.data = np.vstack([self.data,
                                               [iteration, x2, v2, v1, v2 - v1, s, self.h, self.c1, self.c2, ui,
                                                abs(ui - v2)]])
                    else:
                        self.data = np.vstack([self.data,
                                               [iteration, x2, v2, v1, v2 - v1, s, self.h, self.c1, self.c2]])

                    # Заканчиваем вычисления в случае, если находимся в эпсилон окрестности b
                    if abs(x2 - self.b) < EPS:
                        print(f"Вычисление остановилось на вычислении {iteration}")
                        break

                    # Прогнозируем последний шаг
                    if x2+self.h > self.b + EPS:
                        self.h = self.b+EPS-x2

                    self.u = v2
                    self.x = x2
                elif s < self.eps / (2 ** (P + 1)):
                    if self.category == 0:
                        ui = self.true_solve_test(x1)
                        self.data = np.vstack([self.data,
                                               [iteration, x2, v2, v1, v2 - v1, s, self.h, self.c1, self.c2, ui,
                                                abs(ui - v2)]])
                    else:
                        self.data = np.vstack([self.data,
                                               [iteration, x2, v2, v1, v2 - v1, s, self.h, self.c1, self.c2]])
                    self.h = 2 * self.h
                    if abs(x2 - self.b) < EPS:
                        print(f"Вычисление остановилось на вычислении {iteration}")
                        break

                    if x2 + self.h > self.b + EPS:
                        self.h = self.b + EPS - x2

                    self.c1 += 1
                    self.u = v2
                    self.x = x2
                elif s > self.eps:
                    self.h = self.h / 2
                    self.c2 += 1
                    iteration -= 1  # точно уменьшается на 1? Надо проверить

                # if abs(v1 - prev_v) < self.eps:
                #     print(f"Решение стабилизировалось на шаге {iteration}")
                #     break

                prev_v = v1
            return [self.u, self.x]
        else:
            for iteration in range(0, self.n_max):
                x, v = self.__solve_with_lp()
                if abs(x - self.b) < EPS:
                    print(f"Вычисление остановилось на вычислении {iteration}")
                    break

                if x + self.h > self.b + EPS:
                    self.h = self.b + EPS - x
                self.u = v
                self.x = x
                prev_v = v
            return [self.u, self.x]

    def true_solve_test(self, x):  # истинное решение для тестовой задачи
        u = self.u0 * np.exp(-0.5 * x)
        return u

    def plot_solution(self):
        """
        Функция для построения графиков решений:
        - При category == 0: приближенное решение и истинное решение.
        - При category == 1: только приближенное решение.
        """
        np.set_printoptions(threshold=np.inf)
        x_values = self.data[:, 1]  # Значения x
        approx_values = self.data[:, 3]  # Приближенные значения v2
        plt.figure(figsize=(10, 6))
        if self.category == 0:
            true_values = self.data[:, 9]  # Истинные значения ui
            plt.plot(x_values, true_values, label='Истинное решение', color='blue', linestyle='--', linewidth=1)
            plt.plot(x_values, approx_values, label='Приближенное решение', color='red', linestyle='--', linewidth=1)
            plt.title('Сравнение приближенного и истинного решения')
        else:
            plt.plot(x_values, approx_values, label='Приближенное решение', color='green')
            plt.title('Приближенное решение')

        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.legend()
        plt.grid(True)
        plt.show()

# Работа со второй задачей (ДУ второго порядка)
class two_d_diffur:
    def __init__(self, u0: float, z0: float, x0: float, a: float, n_max: int, eps: float, b: float):
        # self.u0 = u0
        # self.x0 = x0
        # self.z0 = z0
        self.u = u0
        self.z = z0
        self.x = x0
        self.a = a
        self.n_max = n_max
        self.eps = eps
        self.h = STEP
        self.c1 = 0
        self.c2 = 0
        self.b = b
        self.data = np.empty((0, 13))
        # [index, xi, vi[0], vi[1], v2i[0], v2i[1], vi[0]-v2i[0], vi[1]-v2i[1], ОЛП[0], ОЛП[1], hi, C1, C2]

    def __f1(self, x, u, z):
        return z

    def __f2(self, x, u, z):
        return -self.a * np.sin(u)

    def __solve_without_lp(self):
        x = self.x
        v = [self.u, self.z]

        k1 = [self.__f1(x, v[0], v[1]),
              self.__f2(x, v[0], v[1])]
        k2 = [self.__f1(x + 0.5 * self.h, v[0] + 0.5 * self.h * k1[0], v[1] + 0.5 * self.h * k1[1]),
              self.__f2(x + 0.5 * self.h, v[0] + 0.5 * self.h * k1[0], v[1] + 0.5 * self.h * k1[1])]
        k3 = [self.__f1(x + 0.5 * self.h, v[0] + 0.5 * self.h * k2[0], v[1] + 0.5 * self.h * k2[1]),
              self.__f2(x + 0.5 * self.h, v[0] + 0.5 * self.h * k2[0], v[1] + 0.5 * self.h * k2[1])]
        k4 = [self.__f1(x + self.h, v[0] + self.h * k3[0], v[1] + self.h * k3[1]),
              self.__f2(x + self.h, v[0] + self.h * k3[0], v[1] + self.h * k3[1])]

        v = [v[0] + 1 / 6 * self.h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
             v[1] + 1 / 6 * self.h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])]
        x = x + self.h

        return x, v

    def __search_k_for_solve_with_lp(self, x, v):
        k1 = [self.__f1(x, v[0], v[1]),
              self.__f2(x, v[0], v[1])]
        k2 = [self.__f1(x + 0.25 * self.h, v[0] + 0.25 * self.h * k1[0], v[1] + 0.25 * self.h * k1[1]),
              self.__f2(x + 0.25 * self.h, v[0] + 0.25 * self.h * k1[0], v[1] + 0.25 * self.h * k1[1])]
        k3 = [self.__f1(x + 0.5 * self.h, v[0] + 0.25 * self.h * k2[0], v[1] + 0.25 * self.h * k2[1]),
              self.__f2(x + 0.5 * self.h, v[0] + 0.25 * self.h * k2[0], v[1] + 0.25 * self.h * k2[1])]
        k4 = [self.__f1(x + 0.5 * self.h, v[0] + 0.5 * self.h * k3[0], v[1] + 0.5 * self.h * k3[1]),
              self.__f2(x + 0.5 * self.h, v[0] + 0.5 * self.h * k3[0], v[1] + 0.5 * self.h * k3[1])]

        return k1, k2, k3, k4

    def __solve_with_lp(self):
        x = self.x
        v = [self.u, self.z]
        # первая часть для 1/2 STEP
        k1, k2, k3, k4 = self.__search_k_for_solve_with_lp(x, v)
        v1 = [v[0] + 1 / 12 * self.h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
              v[1] + 1 / 12 * self.h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])]
        x = x + 0.5 * self.h
        # вторая часть для еще одного 1/2 STEP
        k1, k2, k3, k4 = self.__search_k_for_solve_with_lp(x, v)
        v = [v1[0] + 1 / 12 * self.h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
             v1[1] + 1 / 12 * self.h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])]

        return x, v

    def solve(self, with_lp: bool):
        prev_v = self.u
        if with_lp:
            for iteration in range(0, self.n_max):
                x1, v1 = self.__solve_with_lp()
                x2, v2 = self.__solve_without_lp()
                s = [(v1[0] - v2[0]) / (2 ** P - 1),
                     (v1[1] - v2[1]) / (2 ** P - 1)]
                s_norma = s[0] + s[1]  # ||s||1
                # s_norma_inf = max(s[0], s[1])  #||s||inf

                if self.eps >= s_norma >= self.eps / (2 ** (P + 1)):
                    self.data = np.vstack([self.data,
                                           [iteration, x2, v2[0], v2[1], v1[0], v1[1],
                                            v2[0] - v1[0], v2[1] - v1[1], s[0], s[1], self.h, self.c1, self.c2]])
                    if abs(x2 - self.b) < EPS:
                        print(f"Вычисление остановилось на вычислении {iteration}")
                        break

                    if x2 + self.h > self.b + EPS:
                        self.h = self.b + EPS - x2
                    self.u = v2[0]
                    self.z = v2[1]
                    self.x = x2

                elif s_norma < self.eps / (2 ** (P + 1)):
                    self.data = np.vstack([self.data,
                                           [iteration, x2, v2[0], v2[1], v1[0], v1[1],
                                            v2[0] - v1[0], v2[1] - v1[1], s[0], s[1], self.h, self.c1, self.c2]])
                    self.h = 2 * self.h
                    if abs(x2 - self.b) < EPS:
                        print(f"Вычисление остановилось на вычислении {iteration}")
                        break

                    if x2 + self.h > self.b + EPS:
                        self.h = self.b + EPS - x2
                    self.c1 += 1
                    self.u = v2[0]
                    self.z = v2[1]
                    self.x = x2
                elif s_norma > self.eps:
                    self.h = self.h / 2
                    self.c2 += 1
                    iteration -= 1  # точно уменьшается на 1? Надо проверить
                prev_v = v1
        else:
            for iteration in range(0, self.n_max):
                x, v = self.__solve_with_lp()
                if abs(x - self.b) < EPS:
                    print(f"Вычисление остановилось на вычислении {iteration}")
                    break

                if x + self.h > self.b + EPS:
                    self.h = self.b + EPS - x
                self.u = v[0]
                self.z = v[1]
                self.x = x
                prev_v = v


    def plot_solution(self):
        """
        Функция для построения графиков решений:
        """
        x_values = self.data[:, 1]  # Значения x
        approx_values_0 = self.data[:, 4]  # Приближенные значения v2[0]
        approx_values_1 = self.data[:, 5]  # Приближенные значения v2[1]
        plt.figure(figsize=(10, 6))

        plt.plot(x_values, approx_values_0, label='Приближенное решение 1', color='green')
        plt.title('Приближенное решение 1')

        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))

        plt.plot(x_values, approx_values_1, label='Приближенное решение 2', color='green')
        plt.title('Приближенное решение 2')

        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def PlotPhasePortait(self):
        # Построение фазового портрета
        solution = self.__solve_without_lp()
        plt.figure(figsize=(10, 6))
        u_values = self.data[:, 2]  # Значения u
        z_values = self.data[:, 3]  # Значения u'
        plt.plot(u_values, z_values, label="Фазовый портрет", color="blue")
        plt.title("Фазовый портрет системы")
        plt.xlabel("u")
        plt.ylabel("u'")
        plt.grid(True)
        plt.legend()
        plt.show()

# Примеры использования:

# # Создаем объект класса one_d_diffur, например:
# solver = one_d_diffur(category=0, variant_num=1, u0=0.1, x0=0.0, n_max=500, eps=1e-5)
# solver.solve(with_lp=True)
# solver.plot_solution()

# Решение диффура второго порядка
# solver = two_d_diffur(1, 1, 1, 2, 1500, 1e-5, 6.3)
# solver.solve(with_lp=True)
# solver.plot_solution()
# solver.PlotPhasePortait()

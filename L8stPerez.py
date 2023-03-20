import numpy as np


def fun(a, t):
    return np.array([0 if i <= 0 else i if i <= t else t for i in a])


x = np.array([[1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1],
              [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
              [1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1]])
x0 = np.array([[1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1],
               [-1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1],
               [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]])
w = 1 / 2 * x
w = np.array(w)
print("Матрица весовых коэффициентов:")
print(w)
T = len(x[0]) / 2
E_max = 0.1
e = np.eye(len(x)) - 1 / len(x)
for i in range(len(e)):
    e[i][i] = 1
print("Матрица весов обратных связей")
print(e)

for i in range(len(x0)):
    print("Образ", len(x) + 1 + i)
    count = 1
    X = x0[i]
    s1 = np.dot(w, X) + T
    y1 = fun(s1, T)
    print("Выходные значения нейронов первого слоя:")
    print(y1)
    y2p = y1.copy()

    while True:
        print("Итерация", count)
        print("Выходные значения нейронов второго слоя:")
        s2 = np.dot(e, y2p)
        y2 = fun(s2, T)
        print(y2)
        print("Условие стабилизации:", round(sum((y2 - y2p) ** 2), 3))
        if sum((y2 - y2p) ** 2) <= E_max:
            break
        y2p = y2.copy()
        count += 1

    count = 0
    index = 0
    for i, n in enumerate(y2):
        if n != 0:
            count += 1
            index = i
    if count == 1:
        print("Образ классифицирован к классу", index + 1)
    else:
        print("Образ не классифицирован")

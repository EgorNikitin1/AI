import numpy as np


def chunked(n, lst):
    new = []
    for i in range(len(lst)):
        if i % n == 0:
            new.append([lst[i]])
        else:
            new[-1].append(lst[i])
    return new


x = np.array([[-2.0],
              [-1.5],
              [-1.0],
              [-0.5],
              [0.0],
              [0.5],
              [1.0],
              [1.5],
              [2.0]], float)
y = np.array([[-0.48],
              [-0.78],
              [-0.83],
              [-0.67],
              [-0.20],
              [0.70],
              [1.48],
              [1.17],
              [0.20]], float)
c = np.array([[-2.0],
              [-1.0],
              [0.0],
              [1.0],
              [2.0]], float)
# x = np.array([[0.0],
#               [1.0],
#               [2.0],
#               [3.0],
#               [4.0],
#               [5.0],
#               [6.0],
#               [7.0],
#               [8.0]], float)
# y = np.array([[-2.5],
#               [-1.7],
#               [-1.1],
#               [-0.4],
#               [0.1],
#               [0.4],
#               [0.9],
#               [1.5],
#               [2.2]], float)
# c = np.array([[0.0],
#               [2.0],
#               [4.0],
#               [6.0],
#               [8.0]], float)
r = 1.5
a = 1 / (2 * r ** 2)
h = np.array([])
for i in range(len(x)):
    temp = []
    for j in range(len(c)):
        temp.append(np.exp(-a * (x[i] - c[j]) ** 2))
    h = np.append(h, temp)
h = np.array(chunked(len(c), h.tolist()))
w = np.dot(np.dot(np.linalg.inv(np.dot(h.T, h)), h.T), y)
print("Синаптические коэффициенты:")
print(w)

xt = 1
yt = sum(([np.exp(-a * (xt - c[j]) ** 2) for j in range(len(c))] * w))
print(f"При x = {xt}, y = {round(*yt, 3)}")

error = []
for i in range(len(x)):
    yr = sum(([np.exp(-a * (x[i] - c[j]) ** 2) for j in range(len(c))] * w))
    error.append(abs(y[i] - yr))
err = sum(error) / len(error) * 100
print("Средняя относительная ошибка аппроксимации: ", round(*err, 1), "%", sep="")

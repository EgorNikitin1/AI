import numpy as np


def fun(a):
    for i in range(len(a)):
        if a[i] < 0:
            a[i] = -1
        else:
            a[i] = 1
    return a


x = np.array([[1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1],
              [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1]])
x0 = np.array([1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1])
y = np.array([[1, 1, 1, 1, -1, 1, 1, -1, 1],
              [1, -1, 1, 1, 1, -1, 1, -1, 1]])
w = np.dot(x.T, y)
print("Матрица весовых коэффициентов:")
print(w)
X = x0
count = 0

while True:
    count += 1
    print("Итерация", count)
    temp1 = []
    temp2 = []

    s1 = np.dot(w.T, X)
    print(s1)
    s1 = fun(s1)
    temp1.append(s1)
    temp1 = temp1[0]
    print(temp1)
    temp1_old = temp1
    temp1 = np.array(temp1)

    s2 = np.dot(w, temp1.T)
    print(s2.T)
    s2 = fun(s2)
    temp2.append(s2)
    temp2 = temp2[0]
    temp2 = np.array(temp2)
    print(temp2)

    X = temp2
    if sum((temp1 - temp1_old) ** 2) == 0 and count > 1:
        print("Сеть инициализирована")
        break

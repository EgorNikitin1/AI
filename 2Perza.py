import numpy as np

#Объявление функций
def sigmoid(a, s):
    return 1 / (1 + np.exp(-a * s))

def minv(v):
    return [min(i) for i in zip(*v)]

def maxv(v):
    return [max(i) for i in zip(*v)]

def normalize(v, minv, maxv):
    for i in range(len(v)):
        for j in range(len(v[i])):
            v[i][j] = (v[i][j] - minv[j]) / (maxv[j] - minv[j])
    return v

def chunked(n, lst):
    new = []
    for i in range(len(lst)):
        if i % n == 0:
            new.append([lst[i]])
        else:
            new[-1].append(lst[i])
    return new

#Блок объявления переменных
x = np.array([[-0.12, 0.42, 0.52, 0.17, -0.64, -1.89],
              [0.97, 0.66, 0.47, 0.37, 0.39, 0.51],
              [-2.60, -1.08, -0.24, -0.10, -0.65, -1.90],
              [-1.45, -0.61, -0.41, -0.86, -1.96, -3.70],
              [0.55, 0.97, 1.11, 0.97, 0.55, -0.15],
              [0.54, 0.53, 0.57, 0.64, 0.75, 0.90],
              [0.93, 0.93, 1.11, 1.47, 2.00, 2.72],
              [-4.07, -2.36, -1.46, -1.36, -2.06, -3.56],
              [3.66, 2.25, 1.73, 2.08, 3.31, 5.42],
              [4.87, 2.25, 0.29, -1.02, -1.67, -1.67],
              [0.82, 0.79, 0.76, 0.75, 0.74, 0.75],
              [4.29, 2.19, 0.62, -0.41, -0.91, -0.88],
              [-1.06, 0.91, 1.89, 1.86, 0.83, -1.20],
              [-1.11, -0.49, -0.46, -1.03, -2.20, -3.97],
              [0.71, -0.31, -0.85, -0.92, -0.51, 0.37],
              [-0.93, -0.63, -0.43, -0.35, -0.36, -0.48],
              [-4.78, -2.71, -1.26, -0.46, -0.29, -0.75],
              [-0.42, -0.41, -1.16, -2.67, -4.94, -7.97],
              [3.25, 2.21, 1.60, 1.42, 1.67, 2.35],
              [-2.07, -1.40, -0.95, -0.72, -0.71, -0.92],
              [4.35, 1.59, -0.26, -1.20, -1.22, -0.32],
              [0.68, 0.13, 0.04, 0.42, 1.27, 2.58],
              [-1.63, -1.58, -2.42, -4.16, -6.79, -10.31],
              [0.97, 0.42, 0.07, -0.09, -0.05, 0.18],
              [-0.38, -0.47, -0.12, 0.65, 1.85, 3.48],
              [-2.19, -1.51, -1.44, -2.00, -3.16, -4.95],
              [-1.16, -1.12, -1.16, -1.29, -1.49, -1.77],
              [-0.80, -0.31, 0.02, 0.19, 0.20, 0.05],
              [-1.97, -0.74, -0.04, 0.13, -0.22, -1.10],
              [1.67, 1.86, 1.74, 1.31, 0.58, -0.46],
              [7.55, 4.56, 2.35, 0.93, 0.30, 0.46],
              [-0.79, -0.70, -0.69, -0.77, -0.93, -1.17],
              [4.14, 1.60, -0.24, -1.38, -1.83, -1.58],
              [-0.91, -1.69, -1.64, -0.75, 0.97, 3.52],
              [-4.79, 2.04, -0.08, 1.09, 1.48, 1.07],
              [-3.02, -2.26, -1.75, -1.51, -1.51, -1.77],
              [4.36, 2.67, 1.82, 1.82, 2.66, 4.35],
              [2.23, 1.91, 2.14, 2.92, 4.25, 6.13],
              [2.47, 0.50, -0.56, -0.70, 0.06, 1.73],
              [0.36, -0.39, -0.59, -0.25, 0.63, 2.05],
              [-2.04, -0.29, 0.53, 0.42, -0.61, -2.57],
              [0.00, 0.16, -0.13, -0.88, -2.09, -3.76],
              [-2.75, -1.95, -1.40, -1.08, -0.99, -1.15],
              [0.64, 0.17, -0.03, 0.04, 0.37, 0.97],
              [-0.78, -1.06, -0.87, -0.22, 0.89, 2.48],
              [-0.90, -1.00, -0.98, -0.82, -0.54, -0.13],
              [0.33, 1.14, 1.60, 1.71, 1.46, 0.86],
              [2.20, 1.84, 1.98, 2.62, 3.75, 5.37]], float)
y = np.array([[0, 1],
              [1, 0],
              [0, 1],
              [0, 1],
              [0, 1],
              [1, 0],
              [1, 0],
              [0, 1],
              [1, 0],
              [1, 0],
              [1, 0],
              [1, 0],
              [0, 1],
              [0, 1],
              [1, 0],
              [0, 1],
              [0, 1],
              [0, 1],
              [1, 0],
              [0, 1],
              [1, 0],
              [1, 0],
              [0, 1],
              [1, 0],
              [1, 0],
              [0, 1],
              [0, 1],
              [0, 1],
              [0, 1],
              [0, 1],
              [1, 0],
              [0, 1],
              [1, 0],
              [1, 0],
              [0, 1],
              [0, 1],
              [1, 0],
              [1, 0],
              [1, 0],
              [1, 0],
              [0, 1],
              [0, 1],
              [0, 1],
              [1, 0],
              [1, 0],
              [1, 0],
              [0, 1],
              [1, 0]])
#np.random.seed(1)
#w0 = np.random.random((len(x[0])))
#w = 2 * np.random.random((len(x[0]), len(y[0]))) - 1
w0 = np.array([0.03, 0.02, 0.04, 0.07])
w1 = np.array([[-0.08, -0.06, 0.02, 0.02, -0.08, -0.07],
               [-0.05, -0.01, 0.05, -0.04, -0.01, 0.01]])
w2 = np.array([[0.01, 0.04],
               [0.02, 0.01]])
v = 0.5
a = 2
epoch_quantity = 200
print("Случайные весовые параметры:")
print(w0, w1, w2, sep="\n")

#Нормализация
x = normalize(x, minv(x), maxv(x))
print("Нормализованный x:")
print(x)

#Обучение
print("Результаты:")
for epoch in range(1, epoch_quantity + 1):
    print(f"    Эпоха {epoch}:")
    yr1, yr2, yr1p, yr2p = np.array([]), np.array([]), np.array([]), np.array([])
    sigma1, sigma2 = np.array([]), np.array([])
    err = np.array([])
    for i in range(len(x)):
        temp_y1, temp_y1p  = np.array([]), np.array([])
        temp_y1 = np.append(temp_y1, sigmoid(a, np.dot(x[i], w1.T) + w0[:2]))
        temp_y1p = a * temp_y1 * (1 - temp_y1)
        yr1, yr1p = np.append(yr1, temp_y1), np.append(yr1p, temp_y1p)

        temp_y2, temp_y2p = np.array([]), np.array([])
        temp_y2 = np.append(temp_y2, sigmoid(a, np.dot(temp_y1, w2.T) + w0[2:]))
        temp_y2p = a * temp_y2 * (1 - temp_y2)
        yr2, yr2p = np.append(yr2, temp_y2), np.append(yr2p, temp_y2p)

        delta = np.subtract(y[i], temp_y2)

        temp_sigma1, temp_sigma2 = np.array([0, 0], float), np.array([])
        temp_sigma2 = delta * temp_y2p
        temp_sigma1[0] = (w2[0][0] * temp_sigma2[0] + w2[1][0] * temp_sigma2[1]) * temp_y1p[0]
        temp_sigma1[1] = (w2[0][1] * temp_sigma2[0] + w2[1][1] * temp_sigma2[1]) * temp_y1p[1]
        sigma1 = np.append(sigma1, temp_sigma1)
        sigma2 = np.append(sigma2, temp_sigma2)

        w0[:2] += v * temp_sigma1
        w0[2:] += v * temp_sigma2
        temp_w2w = np.array([])
        for j in range(len(temp_y2)):
            temp_w2 = np.array([])
            temp_w2 = np.add(w2[j], v * temp_y1p * temp_sigma2[j])
            temp_w2w = np.append(temp_w2w, temp_w2)
        w2 = np.array(chunked(2, temp_w2w.tolist()))
        temp_w1w = np.array([])
        for j in range(len(temp_y1)):
            temp_w1 = np.array([])
            temp_w1 = np.add(w1[j], v * x[i] * temp_sigma1[j])
            temp_w1w = np.append(temp_w1w, temp_w1)
        w1 = np.array(chunked(6, temp_w1w.tolist()))
        err = np.append(err, sum(delta ** 2))
    yr1 = np.array(chunked(2, yr1.tolist()))
    yr2 = np.array(chunked(2, yr2.tolist()))
    print("Расчетные данные:")
    #print(yr1)
    #print(yr2)
    print("Общая ошибка:")
    print((sum(err) / (len(y[0]) * len(y))) ** 0.5)
    print(w0, w1, w2, sep="\n")

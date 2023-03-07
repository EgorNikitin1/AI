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

def chunked(lst):
    new = []
    for i in range(len(lst)):
        if i % 2 == 0:
            new.append([lst[i]])
        else:
            new[-1].append(lst[i])
    return new

#Блок объявления переменных
x = np.array([[-5, 2],
              [0, 5],
              [2, -4],
              [-3, 1],
              [5, 0],
              [1, -5],
              [-3, -1],
              [2, 5],
              [4, 3],
              [0, -2]], float)
y = np.array([[-18, -1],
              [-9, 16],
              [15, -18],
              [-10, -3],
              [16, 1],
              [14, -23],
              [-6, -11],
              [-3, 18],
              [7, 12],
              [5, -12]], float)
#np.random.seed(1)
#w0 = np.random.random((len(x[0])))
#w = 2 * np.random.random((len(x[0]), len(y[0]))) - 1
w0 = np.array([0.0, 0.2])
w = np.array([[-0.4, -0.1],
              [0.3, 0.2]])
v = 0.9
epoch_quantity = 25
print("Случайные весовые параметры:")
print(w0)
print(w)

#Нормализация
x = normalize(x, minv(x), maxv(x))
y = normalize(y, minv(y), maxv(y))
print("Нормализованный x:")
print(x)
print("Нормализованный y:")
print(y)


#Обучение
print("Результаты:")
for epoch in range(epoch_quantity + 1):
    if epoch > 0:
        print(f"    Эпоха {epoch}:")
    yr = np.array([])
    err = np.array([])
    for i in range(len(x)):
        temp_y = np.array([])
        temp_y = np.append(temp_y, sigmoid(1, np.dot(x[i], w) + w0))
        delta = np.subtract(y[i], temp_y)
        w0 = w0 + delta * v
        temp_ww = np.array([])
        for j in range(len(x[i])):
            temp_w = np.array([])
            temp_w = np.add(w[j], delta * v * x[i][j])
            temp_ww = np.append(temp_ww, temp_w)
        w = np.array(chunked(temp_ww.tolist()))
        yr = np.append(yr, temp_y)
        err = np.append(err, sum(delta ** 2))
    yr = np.array(chunked(yr.tolist()))
    print("Расчетные данные:")
    #print(yr)
    print("Общая ошибка:")
    print((sum(err) / (len(y[0]) * len(x))) ** 0.5)

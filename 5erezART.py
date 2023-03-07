import numpy as np

x = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1],
              [0, 1, 0, 1, 0, 1, 0, 1, 0],
              [1, 0, 1, 0, 1, 0, 1, 0, 0],
              [0, 1, 1, 1, 0, 1, 0, 1, 0]])
Rkr = 0.7
la = 2
v = 0.6
w, t, y = [], [], [0]
print("Образ 1")
for i in range(1, len(x)+1):
    if max(y) == 0:
        w.append((la * x[i-1]) / (la - 1 + sum(x[i-1])))
        t.append(x[i-1])
        w, t = np.array(w), np.array(t, float)
        print("Синапсы кратковременной памяти:", w, sep="\n")
        print("Синапсы долговременной памяти:", t, sep="\n")
        w, t = w.tolist(), t.tolist()
    else:
        R = []
        for j in range(len(w)):
            R.append(sum(t[j] * x[i-1]) / sum(x[i-1]))
            if R[j] > Rkr:
                winj = j
                print("Критерий R =", R[j], "больше Rкр =", Rkr)
        w, t = np.array(w), np.array(t, float)
        w[winj] = (1 - v) * w[winj] + v * ((la * x[i-1]) / (la - 1 + sum(x[i-1])))
        t[winj] = (1 - v) * t[winj] + v * x[i-1]
        print("Синапсы кратковременной памяти:", w, sep="\n")
        print("Синапсы долговременной памяти:", t, sep="\n")
        w, t = w.tolist(), t.tolist()
    if i < len(x):
        print("Образ", i+1)
        y = []
        for j in range(len(w)):
            y.append(sum(w[j] * x[i]))
        print("y равен:", y)
        if max(y) == 0:
            print("Сходства нет, инициализируем как новый образ:")
        else:
            print("Сходство с образом", y.index(max(y)) + 1)

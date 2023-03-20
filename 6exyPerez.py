import numpy as np

x = np.array([[1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1],
              [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
              [1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1]])
example = np.array([[1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1],
                    [-1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1],
                    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]])
w = np.dot(x.T, x)
for i in range(len(w)):
    w[i][i] = 0
print("Матрица весовых коэффициентов:")
print(w)

for i in range(len(example)):
    y = example[i]
    print("Образ", i + 1 + len(x))
    print("Входной вектор:")
    print(y)
    temp = []
    count = 0
    while True:
        count += 1
        s = np.dot(w, y)
        for j in range(len(s)):
            if s[j] < 0:
                s[j] = -1
            else:
                s[j] = 1
        temp.append(s)
        print("Выходной вектор:")
        for j in range(len(temp)):
            print(temp[j])
        print("Первое условие:", sum((y - s) ** 2))
        if sum((y - s) ** 2) == 0:
            for j in range(len(x)):
                if temp[-1].tolist() == x[j].tolist():
                    print("Образ идентифицирован и соответствует эталонному образу", j + 1)
            break
        if len(temp) >= 2:
            print("Второе условие:", sum((y - temp[1]) ** 2))
        if len(temp) == 3:
            print("Третье условие:", sum((s - temp[2]) ** 2))
            if sum((y - temp[1]) ** 2) == 0 and sum((s - temp[2]) ** 2) == 0:
                print("Образ не идентифицирован")
                break
            del temp[0]
        y = s
    print("Количество итераций:", count)

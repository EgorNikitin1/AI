import numpy as np

x = np.array([[1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
              [1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1]])
example = np.array([[1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
                    [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1]])
w = np.dot(x.T, x)
for i in range(len(w)):
    w[i][i] = 0
print("Синоптические коэффициенты:")
print(w)

y = example[0]

while True:
    s = np.dot(w, y)
    for i in range(len(s)):
        if s[i] < 0:
            s[i] = -1
        else:
            s[i] = 1
    print(s)
    print(sum((y - s) ** 2))
    if sum((y-s)**2) == 0:
        break
    y = s

y = example[1]

temp = []
while True:
    s = np.dot(w, y)
    for i in range(len(s)):
        if s[i] < 0:
            s[i] = -1
        else:
            s[i] = 1

    temp.append(s)

    print(*temp)
    if len(temp) == 3:
        print(sum((y - temp[1]) ** 2), sum((s - temp[2]) ** 2))
        if sum((y - temp[1]) ** 2) == 0 and sum((s - temp[2]) ** 2) == 0:
            break
        del temp[0]

    if sum((y - s) ** 2) == 0:
        print(sum((y - s) ** 2))
        break
    y = s

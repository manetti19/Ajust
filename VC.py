import numpy as np

medicoes = (
    (1, 2),
    (2, 2.5),
    (3, 2),
    (4, 3),
    (5, 3.5),
    (6, 4.5),
    (7, 6),
    (8, 7),
)

A = np.zeros((8, 3), dtype=float)  # 8 linhas, 3 colunas de zeros

# LV = a*LH**2 + b*LH + c

b = np.zeros((8, 1), dtype=float)  # 8 linhas, 1 coluna de zeros

for i in range(0, 8):
    A[i][0] = medicoes[i][0]**2
    A[i][1] = medicoes[i][0]
    A[i][2] = 1
    b[i] = medicoes[i][1]

print(A)
print(b)

AT = A.T

X = np.linalg.solve(AT @ A, AT @ b)

v = A @ X - b

# Coeficientes do ajuste (a, b, c)
a_, b_, c_ = X.flatten()                  # X é 3x1; vira 1D

print(f"Parametros da parabola: a = {a_:.6f}, b = {b_:.6f}, c = {c_:.6f}")

print(v)


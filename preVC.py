import numpy as np

# Dados: (n, (Hg, Hm, Hs), (Vg, Vm, Vs))
medicoes = (
    (1,  (59, 34, 17.0), (40, 14, 23.0)),
    (2,  (60, 44, 53.0), (40,  0,  2.5)),
    (3,  (61, 41, 58.0), (39, 52, 30.0)),
    (4,  (62, 12, 53.0), (39, 48, 41.0)),
    (5,  (62, 41, 52.0), (39, 45,  4.0)),
    (6,  (63, 26, 59.0), (39, 40, 52.0)),
    (7,  (64, 18, 45.0), (39, 37, 15.0)),
    (8,  (64, 54,  1.0), (39, 35, 12.0)),
    (9,  (65, 43, 13.0), (39, 33, 37.0)),
    (10, (66, 12, 45.0), (39, 32, 45.0)),
    (11, (66, 42, 11.0), (39, 33,  9.0)),
    (12, (67,  5, 38.0), (39, 33, 17.0)),
    (13, (67, 31, 57.0), (39, 34,  0.0)),
    (14, (68,  0, 53.0), (39, 34, 42.0)),
    (15, (68, 25, 45.0), (39, 36,  5.0)),
    (16, (68, 56, 11.0), (39, 38,  2.0)),
    (17, (69, 20, 40.0), (39, 39, 38.0)),
    (18, (69, 44, 33.0), (39, 41, 10.0)),
    (19, (70, 18,  3.0), (39, 44,  8.0)),
    (20, (70, 45, 47.0), (39, 48,  4.0)),
    (21, (71, 29, 19.0), (39, 53,  9.0)),
)

def dms_para_decimal(g, m, s):
    return g + m/60.0 + s/3600.0

def graus_para_dms(dd):
    sinal = -1 if dd < 0 else 1
    dd = abs(dd)
    g = int(dd)
    rem = (dd - g) * 60.0
    m = int(rem)
    s = (rem - m) * 60.0
    g *= sinal
    return g, m, s

# Supondo a lista 'medicoes' já definida como:
# medicoes = [(n, (Hg,Hm,Hs), (Vg,Vm,Vs)), ...]

# Opção 1 — por posição (como a lista está na ordem 1..21)
# idx = 13 - 1
# Hg, Hm, Hs = medicoes[idx][2]   -> 39 34 0.0
# print(Hg, Hm, Hs)

# idx = 13 - 1
# Hg, Hm, Hs = medicoes[idx][1]   -> 67 31 57.0
# print(Hg, Hm, Hs)

# idx = 13 - 1
# a = medicoes[idx][2]   -> (39, 34, 0.0)
# print(a)

# idx = 13 - 1
# b = medicoes[idx][1]   -> (67, 31, 57.0)
# print(b)

novas_medicoes = np.zeros((21, 3), dtype=float)  # 21 linhas, 3 colunas de zeros

for i in range(0, 21):
    Hg, Hm, Hs = medicoes[i][1]
    Vg, Vm, Vs = medicoes[i][2]
    novas_medicoes[i][1] = dms_para_decimal(Hg, Hm, Hs)
    novas_medicoes[i][2] = dms_para_decimal(Vg, Vm, Vs)

# LV = a*LH**2 + b*LH + c

A = np.zeros((21, 3), dtype=float)  # 21 linhas, 3 colunas de zeros
b = np.zeros((21, 1), dtype=float)  # 21 linhas, 1 coluna de zeros

for i in range(0, 21):
    A[i][0] = novas_medicoes[i][1]**2
    A[i][1] = novas_medicoes[i][1]
    A[i][2] = 1
    b[i] = novas_medicoes[i][2]

print(A)
print(b)
print(novas_medicoes[20])
print(novas_medicoes[20][0])
print(A[20])

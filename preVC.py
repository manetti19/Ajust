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

AT = A.T

X = np.linalg.solve(AT @ A, AT @ b)



'''
print(X)

v = A @ X - b
print(v)

S = 0
for i in range(0, 21):
    S = S + v[i]

S = S/21 #media
print(S)

std_v = np.std(v, ddof=1) # ddof=1 eh pra ser o amostral
print(std_v)
var_v = np.var(v) # sem esse ddof=1 nao sera o amostral
print(var_v)
sstdd_v = np.sqrt(var_v)
print(sstdd_v)


                  eliminando outliers         precisa???

for i in range(0, 21):
    a = np.abs(v[i]) - S
    if a > std_v:
        print(i)
        print(v[i])

novo_v = np.delete(v, [0, 1, 18], axis=0) # esse axis=0 eliminou A LINHA, se fosse =1 seria a coluna
print(novo_v)
novo_vT = novo_v.T
print(novo_vT)
k = novo_vT @ novo_v
print(k)
print(novo_v.shape)



S = 0
for i in range(0, 18):
    S = S + v[i]

S = S/18 #media
print(S)

std_v = np.std(novo_v, ddof=1) # ddof=1 eh pra ser o amostral
print(std_v)
var_v = np.var(novo_v) # sem esse ddof=1 nao sera o amostral
print(var_v)
sstdd_v = np.sqrt(var_v)
print(sstdd_v)

for i in range(0, 18):
    a = np.abs(novo_v[i]) - S
    if a > std_v:
        print(i)
        print(novo_v[i])
'''




# Coeficientes do ajuste (a, b, c)
a_, b_, c_ = X.flatten()                  # X é 3x1; vira 1D

print(f"Parametros da parabola: a = {a_:.6f}, b = {b_:.6f}, c = {c_:.6f}")

# Mínimo da parábola (culminação)
LH_min = -b_ / (2*a_)
LV_min = c_ - (b_**2) / (4*a_)


gH, mH, sH = graus_para_dms(LH_min)
gV, mV, sV = graus_para_dms(LV_min)

print(f"L_H* = {LH_min:.6f}°  -> {gH:02d}° {mH:02d}′ {sH:05.2f}″")
print(f"z_min = L_V* = {LV_min:.6f}°  -> {gV:02d}° {mV:02d}′ {sV:05.2f}″")



'''        isso eh do chat
# --- 3) Resíduos e R² (qualidade do ajuste) ---
y_hat = (A @ X).ravel()                   # valores ajustados (1D)
y_obs = b.ravel()                         # observados (1D)
v = y_obs - y_hat                         # resíduos (convém este sinal)

RSS = np.sum(v**2)                        # soma dos quadrados dos resíduos
TSS = np.sum((y_obs - y_obs.mean())**2)   # soma total dos quadrados
R2  = 1 - RSS/TSS

print(f"R² = {R2:.6f}")
print(f"RMSE ≈ {np.sqrt(RSS/(len(y_obs)-3)):.6f} grau")

# --- 4) Estatística simples dos resíduos (1D) ---
print(f"média(v) = {v.mean():.6e} grau")
print(f"std amostral(v) = {np.std(v, ddof=1):.6e} grau")

# --- 5) (Opcional) Incertezas de L_H* e L_V* via propagação ---
ATA_inv = np.linalg.inv(A.T @ A)          # (3x3)
sigma2 = RSS / (len(y_obs) - 3)           # variância residual
Cov = sigma2 * ATA_inv                    # cov(a,b,c)

dx_da =  b_ / (2*a_**2)
dx_db = -1 / (2*a_)
Jx = np.array([dx_da, dx_db, 0.0])
sx = np.sqrt(Jx @ Cov @ Jx)               # σ(L_H*)

dy_da =  (b_**2) / (4*a_**2)
dy_db = -b_ / (2*a_)
dy_dc =  1.0
Jy = np.array([dy_da, dy_db, dy_dc])
sy = np.sqrt(Jy @ Cov @ Jy)               # σ(L_V*) = σ(z_min)

print(f"σ(L_H*) = {sx:.6f}°  ({sx*60:.3f}′; {sx*3600:.2f}″)")
print(f"σ(z_min) = {sy:.6f}°  ({sy*60:.3f}′; {sy*3600:.2f}″)")
'''
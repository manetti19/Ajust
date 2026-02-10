import numpy as np
import pandas as pd
import sys

# Forçar UTF-8 no Windows (evita erro com σ₀²)
sys.stdout.reconfigure(encoding='utf-8')

# ----------------------------------------------------------
# DADOS DE ENTRADA
# ----------------------------------------------------------
HA = 0.0  # ponto de referência

# Cada linha segue o sentido indicado nas setas do esquema:
# (De -> Para)  Desnível (m), Distância (km)
linhas = [
    ['A', 'B', +6.16, 4],  # Linha 1
    ['A', 'C', +12.57, 2], # Linha 2
    ['B', 'C', +6.41, 2],  # Linha 3
    ['A', 'D', +1.09, 4],  # Linha 4
    ['C', 'D', -11.58, 2], # Linha 5 (seta é de D→C, por isso negativo)
    ['B', 'D', -5.07, 4]   # Linha 6 (seta é de D→B, por isso negativo)
]

# ----------------------------------------------------------
# MONTA A MATRIZ A (incidência) E O VETOR L (desníveis)
# ----------------------------------------------------------
# Parâmetros: [HB, HC, HD]
A = []
L = []

for de, para, desnivel, dist in linhas:
    linha = [0, 0, 0]  # [HB, HC, HD]
    # para - de = l_i
    if para == 'B': linha[0] = +1
    if para == 'C': linha[1] = +1
    if para == 'D': linha[2] = +1
    if de == 'B': linha[0] = -1
    if de == 'C': linha[1] = -1
    if de == 'D': linha[2] = -1
    A.append(linha)
    L.append(desnivel)

A = np.array(A, dtype=float)
L = np.array(L, dtype=float).reshape(-1, 1)

# ----------------------------------------------------------
# MATRIZ DE PESOS
# ----------------------------------------------------------
distancias = np.array([linha[3] for linha in linhas])
P = np.diag(1 / distancias)

# ----------------------------------------------------------
# AJUSTAMENTO PARAMÉTRICO
# ----------------------------------------------------------
N = A.T @ P @ A
n = A.T @ P @ L
X = np.linalg.inv(N) @ n
V = A @ X - L

# ----------------------------------------------------------
# RESULTADOS
# ----------------------------------------------------------
HB, HC, HD = X.flatten()

print("===== RESULTADOS DO AJUSTAMENTO =====")
print(f"HB = {HB:.2f} m")
print(f"HC = {HC:.2f} m")
print(f"HD = {HD:.2f} m")

print("\n===== RESÍDUOS (m) =====")
for i, v in enumerate(V.flatten(), start=1):
    print(f"v{i} = {v:.4f}")

# ----------------------------------------------------------
# VARIÂNCIA A POSTERIORI
# ----------------------------------------------------------
n_obs = len(L)
u = 3
gl = n_obs - u
print (f"\nGraus de liberdade = {gl:f}")
sigma0_2 = float((V.T @ P @ V)[0, 0] / (n_obs - u))
print(f"\nVariância da unidade de peso σ₀² = {sigma0_2:.6f}")
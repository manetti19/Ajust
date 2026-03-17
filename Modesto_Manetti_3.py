import numpy as np
import pandas as pd

# ============================================================
# TRANSFORMAÇÃO AFIM: PIXEL -> UTM
# 7 pontos para ajuste e 3 pontos para teste
# ============================================================

# ------------------------------------------------------------
# 1. DADOS
# ------------------------------------------------------------
pontos = [1, 2, 5, 6, 8, 9, 10, 11, 12, 13]

x_pixel = np.array([1208, 1157,  134,  846, 2238,  518, 2418, 2639, 1524, 2678], dtype=float)
y_pixel = np.array([2414, 1735,  353, 1073, 1317, 1641,  244, 2598, 1019, 1050], dtype=float)

E_utm = np.array([
    681100.209,
    681079.091,
    680377.816,
    680888.011,
    681875.866,
    680626.292,
    682031.497,
    682134.395,
    681361.203,
    682208.231
], dtype=float)

N_utm = np.array([
    7464305.984,
    7464791.902,
    7465806.071,
    7465251.245,
    7465066.854,
    7464876.497,
    7465841.798,
    7464127.842,
    7465299.251,
    7465250.939
], dtype=float)

# ------------------------------------------------------------
# 2. SEPARAÇÃO: 7 PARA AJUSTE, 3 PARA TESTE
# ------------------------------------------------------------
idx_ajuste = np.array([0, 1, 2, 3, 4, 5, 6])   # P1, P2, P5, P6, P8, P9, P10
idx_teste  = np.array([7, 8, 9])               # P11, P12, P13

x_aj = x_pixel[idx_ajuste]
y_aj = y_pixel[idx_ajuste]
E_aj = E_utm[idx_ajuste]
N_aj = N_utm[idx_ajuste]

x_te = x_pixel[idx_teste]
y_te = y_pixel[idx_teste]
E_te = E_utm[idx_teste]
N_te = N_utm[idx_teste]

pontos_aj = np.array(pontos)[idx_ajuste]
pontos_te = np.array(pontos)[idx_teste]

# ------------------------------------------------------------
# 3. AJUSTE DA TRANSFORMAÇÃO AFIM
# ------------------------------------------------------------
# Modelo:
# E = a0 + a1*x + a2*y
# N = b0 + b1*x + b2*y

A = np.column_stack((np.ones(len(x_aj)), x_aj, y_aj))

param_E = np.linalg.inv(A.T @ A) @ (A.T @ E_aj)
param_N = np.linalg.inv(A.T @ A) @ (A.T @ N_aj)

a0, a1, a2 = param_E
b0, b1, b2 = param_N

print("=== TRANSFORMAÇÃO AFIM (PIXEL -> UTM) ===")
print(f"E = {a0:.6f} + {a1:.6f}*x + {a2:.6f}*y")
print(f"N = {b0:.6f} + {b1:.6f}*x + {b2:.6f}*y")
print()

# ------------------------------------------------------------
# 4. RESÍDUOS NOS PONTOS DE AJUSTE
# ------------------------------------------------------------
E_calc_aj = a0 + a1 * x_aj + a2 * y_aj
N_calc_aj = b0 + b1 * x_aj + b2 * y_aj

vE_aj = E_aj - E_calc_aj
vN_aj = N_aj - N_calc_aj

RMS_E_aj = np.sqrt(np.mean(vE_aj**2))
RMS_N_aj = np.sqrt(np.mean(vN_aj**2))
RMS_plani_aj = np.sqrt(np.mean(vE_aj**2 + vN_aj**2))

print("=== AJUSTE: 7 PONTOS ===")
print(f"RMS E: {RMS_E_aj:.4f} m")
print(f"RMS N: {RMS_N_aj:.4f} m")
print(f"RMS planimétrico: {RMS_plani_aj:.4f} m")
print()

df_aj = pd.DataFrame({
    "Ponto": pontos_aj,
    "x_pixel": x_aj,
    "y_pixel": y_aj,
    "E_obs": E_aj,
    "N_obs": N_aj,
    "E_calc": E_calc_aj,
    "N_calc": N_calc_aj,
    "vE": vE_aj,
    "vN": vN_aj
})

print("=== RESÍDUOS DOS PONTOS DE AJUSTE ===")
print(df_aj.to_string(index=False))
print()

# ------------------------------------------------------------
# 5. TESTE NOS 3 PONTOS
# ------------------------------------------------------------
E_calc_te = a0 + a1 * x_te + a2 * y_te
N_calc_te = b0 + b1 * x_te + b2 * y_te

vE_te = E_te - E_calc_te
vN_te = N_te - N_calc_te

RMS_E_te = np.sqrt(np.mean(vE_te**2))
RMS_N_te = np.sqrt(np.mean(vN_te**2))
RMS_plani_te = np.sqrt(np.mean(vE_te**2 + vN_te**2))

print("=== TESTE: 3 PONTOS ===")
print(f"RMS E: {RMS_E_te:.4f} m")
print(f"RMS N: {RMS_N_te:.4f} m")
print(f"RMS planimétrico: {RMS_plani_te:.4f} m")
print()

df_te = pd.DataFrame({
    "Ponto": pontos_te,
    "x_pixel": x_te,
    "y_pixel": y_te,
    "E_obs": E_te,
    "N_obs": N_te,
    "E_calc": E_calc_te,
    "N_calc": N_calc_te,
    "vE": vE_te,
    "vN": vN_te
})

print("=== RESULTADOS DOS PONTOS DE TESTE ===")
print(df_te.to_string(index=False))
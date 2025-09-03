# -*- coding: utf-8 -*-
import numpy as np  # biblioteca para álgebra linear

# -----------------------------
# Observações fornecidas
# -----------------------------
L1 = np.array([232.5, 232.7, 232.5, 232.4, 232.4])  # 5 medições do comprimento (m)
L2 = np.array([72.8, 72.6, 72.5, 72.4, 72.7])       # 5 medições da largura (m)

# -----------------------------
# Estatística das médias
# -----------------------------
n = len(L1)                           # número de repetições (assumindo mesmo n em L1 e L2)
l1 = L1.mean()                        # média de l1
l2 = L2.mean()                        # média de l2
s1_sq = L1.var(ddof=1)                # variância amostral de l1 (dividindo por n-1)
s2_sq = L2.var(ddof=1)                # variância amostral de l2
var_l1 = s1_sq / n                    # variância da MÉDIA de l1 (s^2 / n)
var_l2 = s2_sq / n                    # variância da MÉDIA de l2

# Matriz de covariância das observações (assumindo l1 e l2 não correlacionados)
Sigma_L = np.array([[var_l1, 0.0],
                    [0.0,    var_l2]])

# -----------------------------
# Funções e jacobiana
# -----------------------------
d = np.hypot(l1, l2)                  # d = sqrt(l1^2 + l2^2)
A = l1 * l2                           # A = l1 * l2

# Derivadas parciais: J = ∂[d, A]/∂[l1, l2]
J = np.array([[l1/d,  l2/d],          # [∂d/∂l1, ∂d/∂l2]
              [l2,    l1  ]])         # [∂A/∂l1, ∂A/∂l2]

# -----------------------------
# Propagação: Σx = J ΣL J^T
# -----------------------------
Sigma_x = J @ Sigma_L @ J.T

# -----------------------------
# Relatório
# -----------------------------
print(f"médias: l1 = {l1:.3f} m, l2 = {l2:.3f} m")
print(f"d = {d:.3f} m,  A = {A:.3f} m²")
print("\nΣ_L (variâncias das médias) [m²]:")
print(np.array2string(Sigma_L, formatter={'float_kind':lambda x: f"{x:0.6f}"}))
print("\nJacobian J:")
print(np.array2string(J, formatter={'float_kind':lambda x: f"{x:0.6f}"}))
print("\nΣ_x (covariâncias de [d, A]) — unidades:")
print("  Var(d) em m², Var(A) em m⁴, Cov(d,A) em m³")
print(np.array2string(Sigma_x, formatter={'float_kind':lambda x: f"{x:0.9f}"}))

# Desvios-padrão (para referência)
sd_d = np.sqrt(Sigma_x[0,0])          # σ_d  (m)
sd_A = np.sqrt(Sigma_x[1,1])          # σ_A  (m²)
print(f"\nσ_d = {sd_d:.4f} m   |   σ_A = {sd_A:.4f} m²")

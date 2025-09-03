# -*- coding: utf-8 -*-
# Propagação de covariância para X = [X1, X2]:
# X1 = sqrt(L1^2 + L2^2)   (diagonal)
# X2 = L1 * L2             (área)

import numpy as np
from math import sqrt

# ===== 1) Dados de entrada (em metros) =====
L1 = np.array([232.5, 232.7, 232.5, 232.4, 232.4], dtype=float)  # comprimento
L2 = np.array([ 72.8,  72.6,  72.5,  72.4,  72.7], dtype=float)  # largura

# ===== 2) Estatísticas básicas =====
L1_bar = float(L1.mean())
L2_bar = float(L2.mean())

# Matriz de covariância amostral de [L1, L2]^T (não-viesada: divide por n-1)
# Σ_L = [[Var(L1), Cov(L1,L2)],
#        [Cov(L1,L2), Var(L2)]]
Sigma_L = np.cov(np.vstack([L1, L2]), bias=False)  # shape (2,2)

# ===== 3) Jacobiana J em (L1_bar, L2_bar) =====
# X1 = sqrt(L1^2 + L2^2)  -> [∂X1/∂L1, ∂X1/∂L2] = [L1/X1, L2/X1]
# X2 = L1*L2              -> [∂X2/∂L1, ∂X2/∂L2] = [L2, L1]
X1_bar = sqrt(L1_bar**2 + L2_bar**2)
X2_bar = L1_bar*L2_bar
J = np.array([
    [L1_bar / X1_bar, L2_bar / X1_bar],  # derivadas de X1
    [L2_bar,          L1_bar]            # derivadas de X2
], dtype=float)  # shape (2,2)

# ===== 4) Propagação: Σ_X = J Σ_L J^T =====
Sigma_X = J @ Sigma_L @ J.T  # shape (2,2)

# ===== 5) Extras: desvios-padrão e correlação entre X1 e X2 =====
sigma_X1 = float(np.sqrt(Sigma_X[0, 0]))
sigma_X2 = float(np.sqrt(Sigma_X[1, 1]))
rho_X1X2 = float(Sigma_X[0, 1] / (sigma_X1 * sigma_X2)) if sigma_X1 > 0 and sigma_X2 > 0 else np.nan

# ===== 6) Saída =====
np.set_printoptions(precision=6, suppress=True)
print("Médias (m):   L1̄ = %.6f   L2̄ = %.6f   X1 = %.6f   X2 = %.6f" % (L1_bar, L2_bar, X1_bar, X2_bar))
print("\nΣ_L  (m²):")
print(Sigma_L)

print("\nJacobiana J:")
print(J)

print("\nΣ_X = J Σ_L Jᵀ (covariâncias de [X1 (m), X2 (m²)]):")
print(Sigma_X)

print("\nDesvios-padrão:")
print("σ_X1 = %.6f m"   % sigma_X1)
print("σ_X2 = %.6f m²"  % sigma_X2)
print("Correlação ρ(X1, X2) = %.6f" % rho_X1X2)

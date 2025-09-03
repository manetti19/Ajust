# -*- coding: utf-8 -*-

# Importa o NumPy para contas matriciais
import numpy as np

# -----------------------------
# ENTRADAS DO PROBLEMA
# -----------------------------

# Ângulos médios observados (em graus) e lado c (em metros)
A_deg = 90.0          # Ângulo A
B_deg = 45.0          # Ângulo B
c     = 100.0         # Lado c (m)

# Precisões das observações
sigma_A_sec = 4.0     # desvio-padrão de A, em segundos de arco
sigma_B_sec = 4.0     # desvio-padrão de B, em segundos de arco
sigma_c_m   = 0.05    # desvio-padrão de c, em metros

# ÚNICA correlação não nula informada: σ_AB = 1''
# Interpretamos como COV(A,B) = (1'')^2 em unidades rad^2
sigma_AB_sec = 1.0    # "1 segundo de arco" para a covariância angular A-B

# -----------------------------
# CONVERSÕES DE UNIDADES
# -----------------------------

# Fator para converter segundos de arco -> radianos: 1'' = pi / (180*3600) rad
sec_to_rad = np.pi / (180.0 * 3600.0)

# Converte ângulos para radianos
A = np.deg2rad(A_deg)
B = np.deg2rad(B_deg)

# Converte desvios-padrão angulares para radianos
sigma_A = sigma_A_sec * sec_to_rad
sigma_B = sigma_B_sec * sec_to_rad

# Converte a "covariância" fornecida (1'') para rad^2 (assumindo (1'')^2)
cov_AB = (sigma_AB_sec * sec_to_rad) ** 2

# -----------------------------
# FUNÇÕES GEOMÉTRICAS
# -----------------------------

# Para evitar recomputar, definimos S = sin(A+B)
S = np.sin(A + B)

# Lado a pela lei dos senos usando sin(C)=sin(A+B)
a = c * np.sin(A) / S

# Lado b pela lei dos senos usando sin(C)=sin(A+B)
b = c * np.sin(B) / S

# Pacote de resultados (vetor X)
X = np.array([a, b])  # [a, b]

# -----------------------------
# JACOBIANA (derivadas parciais)
# Modelos: 
# a = c * sin(A) / S,  S = sin(A+B)
# b = c * sin(B) / S
# -----------------------------

# Derivadas auxiliares
cosApB = np.cos(A + B)     # cos(A+B)
sinA   = np.sin(A)         # sin(A)
sinB   = np.sin(B)         # sin(B)
cosA   = np.cos(A)         # cos(A)
cosB   = np.cos(B)         # cos(B)
S2     = S**2              # S^2, aparecerá no quociente

# ∂a/∂A = c*(cosA*S - sinA*cos(A+B)) / S^2
daa_dA = c * (cosA * S - sinA * cosApB) / S2

# ∂a/∂B = c*(0*S - sinA*∂S/∂B)/S^2 = -c*sinA*cos(A+B)/S^2
daa_dB = -c * sinA * cosApB / S2

# ∂a/∂c = sinA / S
daa_dc = sinA / S

# ∂b/∂A = -c*sinB*cos(A+B)/S^2
dbb_dA = -c * sinB * cosApB / S2

# ∂b/∂B = c*(cosB*S - sinB*cos(A+B))/S^2
dbb_dB = c * (cosB * S - sinB * cosApB) / S2

# ∂b/∂c = sinB / S
dbb_dc = sinB / S

# Monta a Jacobiana J (2x3) na ordem de L = [A, B, c]
J = np.array([
    [daa_dA, daa_dB, daa_dc],
    [dbb_dA, dbb_dB, dbb_dc]
])

# -----------------------------
# MATRIZ DE COVARIÂNCIAS DAS OBSERVAÇÕES Σ_L
# L = [A, B, c] com unidades (rad, rad, m)
# -----------------------------

Sigma_L = np.array([
    [sigma_A**2,  cov_AB,     0.0        ],
    [cov_AB,      sigma_B**2, 0.0        ],
    [0.0,         0.0,        sigma_c_m**2]
])

# -----------------------------
# PROPAGAÇÃO: Σ_x = J Σ_L J^T
# -----------------------------

Sigma_x = J @ Sigma_L @ J.T

# -----------------------------
# RELATÓRIO DOS RESULTADOS
# -----------------------------

# Desvios-padrão finais (m)
sigma_a = np.sqrt(Sigma_x[0, 0])
sigma_b = np.sqrt(Sigma_x[1, 1])

# Covariância e correlação entre a e b
cov_ab = Sigma_x[0, 1]
rho_ab = cov_ab / (sigma_a * sigma_b)

# Impressão formatada
print("Resultados (valores médios):")
print(f"a = {a:,.3f} m")
print(f"b = {b:,.3f} m")
print()
print("Matriz de covariâncias Σ_x (m^2):")
print(np.array2string(Sigma_x, formatter={'float_kind':lambda x: f"{x:0.8f}"}))
print()
print("Precisões (desvios-padrão):")
print(f"σ_a = {sigma_a*100:.2f} cm")
print(f"σ_b = {sigma_b*100:.2f} cm")
print(f"Cov(a,b) = {cov_ab:0.8f} m^2   |   ρ(a,b) = {rho_ab:0.6f}")
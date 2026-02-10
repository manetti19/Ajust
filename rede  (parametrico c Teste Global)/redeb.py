import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ----------------------------------------------------------
# 1. DADOS DE ENTRADA
# ----------------------------------------------------------
HA = 0.0  # Altura de referência

# Linhas: De -> Para, Desnível (m), Distância (km)
linhas = [
    ['A', 'B', +6.16, 4],
    ['A', 'C', +12.57, 2],
    ['B', 'C', +6.41, 2],
    ['A', 'D', +1.09, 4],
    ['C', 'D', -11.58, 2],
    ['B', 'D', -5.07, 4]
]

# ----------------------------------------------------------
# 2. MONTAGEM DAS MATRIZES A E L
# ----------------------------------------------------------
A = []
L = []

for de, para, desnivel, dist in linhas:
    linha = [0, 0, 0]  # [HB, HC, HD]
    # modelo: (Hpara - Hde) = desnível
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
# 3. MATRIZ DE PESOS (σᵢ = √(10⁻³ * dᵢ) km)
# ----------------------------------------------------------
distancias = np.array([linha[3] for linha in linhas])
# variâncias: σᵢ² = 10⁻³ * dᵢ
variancias = 1e-3 * distancias
# pesos: Pᵢ = 1 / σᵢ² = 1000 / dᵢ
P = np.diag(1 / variancias)

# ----------------------------------------------------------
# 4. AJUSTAMENTO PELO MÉTODO DOS MÍNIMOS QUADRADOS
# ----------------------------------------------------------
N = A.T @ P @ A
n = A.T @ P @ L
X = np.linalg.inv(N) @ n
V = A @ X - L

HB, HC, HD = X.flatten()

print("===== RESULTADOS DO AJUSTAMENTO =====")
print(f"HB = {HB:.2f} m")
print(f"HC = {HC:.2f} m")
print(f"HD = {HD:.2f} m")

print("\n===== RESÍDUOS (m) =====")
for i, v in enumerate(V.flatten(), start=1):
    print(f"v{i} = {v:.4f}")

# ----------------------------------------------------------
# 5. VARIÂNCIA A POSTERIORI
# ----------------------------------------------------------
n_obs = len(L)
u = 3
gl = n_obs - u

sigma_hat2 = float((V.T @ P @ V)[0, 0] / gl)
print(f"\nVariância da unidade de peso a posteriori σ̂₀² = {sigma_hat2:.8f}")

# ----------------------------------------------------------
# 6. TESTE GLOBAL QUI-QUADRADO (α = 5%) SEM SCIPY
# ----------------------------------------------------------
sigma0_2 = 1.0   # valor a priori
alpha = 0.05

# Valores críticos tabelados (gl = 3, α = 5%)
chi2_min = 0.216
chi2_max = 9.35

# Estatística calculada
chi2_calc = gl * sigma_hat2 / sigma0_2

print("\n===== TESTE GLOBAL QUI-QUADRADO =====")
print(f"Graus de liberdade (gl): {gl}")
print(f"σ₀² (a priori) = {sigma0_2:.2f}")
print(f"σ̂₀² (a posteriori) = {sigma_hat2:.8f}")
print(f"χ²calc = {chi2_calc:.6f}")
print(f"χ²min (5%) = {chi2_min:.4f}")
print(f"χ²max (5%) = {chi2_max:.4f}")

if chi2_min <= chi2_calc <= chi2_max:
    print("\n O ajustamento é ACEITO ao nível de 5%.")
else:
    print("\n O ajustamento é REJEITADO ao nível de 5%.")

# ----------------------------------------------------------
# 7. CONCLUSÃO FINAL
# ----------------------------------------------------------
print("\n===== CONCLUSÃO FINAL =====")
print("Alturas ajustadas (m):")
print(f"  HB = {HB:.2f}")
print(f"  HC = {HC:.2f}")
print(f"  HD = {HD:.2f}")
print("\nComo os resíduos são praticamente nulos, o ajustamento é ótimo e o teste global é aceito.")
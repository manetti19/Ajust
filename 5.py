# -*- coding: utf-8 -*-

# -----------------------------
# Entradas do problema
# -----------------------------
a = 30.0        # lado a (m)
b = 40.0        # lado b (m)
C_deg = 60.0    # ângulo C (graus)

# Precisões (desvios-padrão)
sigma_a = 0.003     # ±3 mm -> 0,003 m
sigma_b = 0.004     # ±4 mm -> 0,004 m
sigma_C_arcsec = 5  # ±5'' (segundos de arco)

# -----------------------------
# Conversões
# -----------------------------
import math                               # importa a math para trigonometria
C = math.radians(C_deg)                   # converte C para radianos
sec_to_rad = math.pi / (180.0 * 3600.0)   # fator: 1'' -> rad
sigma_C = sigma_C_arcsec * sec_to_rad     # σ_C em radianos

# -----------------------------
# 1) Estimativa de c pela Lei dos Cossenos
#    c = sqrt(a^2 + b^2 - 2ab cos C)
# -----------------------------
c = math.sqrt(a*a + b*b - 2.0*a*b*math.cos(C))

# -----------------------------
# 2) Jacobiana (derivadas parciais de c)
#    dc/da = (a - b cos C)/c
#    dc/db = (b - a cos C)/c
#    dc/dC = (ab sin C)/c
# -----------------------------
dc_da = (a - b*math.cos(C)) / c
dc_db = (b - a*math.cos(C)) / c
dc_dC = (a*b*math.sin(C)) / c

# -----------------------------
# 3) Propagação de variâncias (sem correlações)
#    Var(c) = (dc/da)^2 σ_a^2 + (dc/db)^2 σ_b^2 + (dc/dC)^2 σ_C^2
# -----------------------------
var_c_a = (dc_da**2) * (sigma_a**2)   # contribuição de a
var_c_b = (dc_db**2) * (sigma_b**2)   # contribuição de b
var_c_C = (dc_dC**2) * (sigma_C**2)   # contribuição de C
var_c = var_c_a + var_c_b + var_c_C   # variância total de c
sigma_c = math.sqrt(var_c)            # desvio-padrão de c

# -----------------------------
# 4) Relatório
# -----------------------------
print("Lei dos Cossenos + Propagação (Jacobiana)")
print("-----------------------------------------")
print(f"a = {a:.3f} m  ± {sigma_a*1000:.1f} mm")
print(f"b = {b:.3f} m  ± {sigma_b*1000:.1f} mm")
print(f"C = {C_deg:.3f}° ± {sigma_C_arcsec:.0f}''")
print()
print(f"c = {c:.6f} m")
print(f"σ_c = {sigma_c:.6f} m  (~{sigma_c*1000:.2f} mm)")
print()
print("Derivadas (sensibilidades):")
print(f"dc/da = {dc_da:.6f},  dc/db = {dc_db:.6f},  dc/dC = {dc_dC:.6f} (m/rad)")
print()
print("Decomposição da variância de c (m²):")
print(f"  de a: {var_c_a:.9f}")
print(f"  de b: {var_c_b:.9f}")
print(f"  de C: {var_c_C:.9f}")
print(f"  total: {var_c:.9f}")

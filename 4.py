# -*- coding: utf-8 -*-
# Propagação-padrão (Jacobiana) para D = sqrt(d^2 + h^2)
# Preencha SOMENTE a seção "SUBSTITUA AQUI" e rode.

import math

# ============================================================
# SUBSTITUA AQUI (valores do seu problema)
# ------------------------------------------------------------
d = 400.0          # distância horizontal (m)
h = 80.0           # altura (m)

sigma_d = 0.000    # desvio-padrão de d (m). Use 0.0 se "desprezível".
sigma_h = None     # desvio-padrão de h (m). Deixe None se quiser resolver a partir de sigma_D_alvo.

rho_dh  = 0.0      # correlação entre d e h. Se desconhecida, use 0.0

sigma_D_alvo = 0.020  # (opcional) desvio-padrão alvo para D (m)  -> 20 mm no enunciado
# ============================================================


# ------------------------------
# 1) Funções utilitárias
# ------------------------------
def jacobiana_D(d: float, h: float):
    """
    Retorna o vetor Jacobiano J = [∂D/∂d, ∂D/∂h] para D = sqrt(d^2 + h^2).
    """
    D = math.hypot(d, h)            # sqrt(d^2 + h^2)
    return (d / D, h / D), D        # (∂D/∂d, ∂D/∂h), D

def sigma_D_por_propagacao(d, h, sigma_d, sigma_h, rho_dh=0.0):
    """
    Calcula sigma_D (m) pela fórmula geral: Var(D) = J Σ_L J^T,
    com Σ_L = [[sigma_d^2, rho*sigma_d*sigma_h], [rho*sigma_d*sigma_h, sigma_h^2]].
    """
    (dDd, dDh), D = jacobiana_D(d, h)           # derivadas e D
    cov_dh = rho_dh * sigma_d * sigma_h         # covariância a partir da correlação
    var_D = (dDd**2) * (sigma_d**2) + (dDh**2) * (sigma_h**2) + 2.0 * dDd * dDh * cov_dh
    return math.sqrt(var_D), D                   # retorna σ_D e D

def resolver_sigma_h_para_sigma_D_alvo(d, h, sigma_d, sigma_D_alvo, rho_dh=0.0):
    """
    Resolve σ_h necessário para atingir σ_D_alvo, mantendo σ_d e ρ fixos.
    Parte da equação:
      σ_D^2 = (d/D)^2 σ_d^2 + (h/D)^2 σ_h^2 + 2 (d/D) (h/D) ρ σ_d σ_h
    Isso é um quadrático em σ_h:  A σ_h^2 + B σ_h + C = 0
      A = (h/D)^2
      B = 2 (d/D)(h/D) ρ σ_d
      C = (d/D)^2 σ_d^2 - σ_D^2
    Pegamos a raiz positiva (física).
    """
    (dDd, dDh), D = jacobiana_D(d, h)
    A = (dDh**2)
    B = 2.0 * dDd * dDh * rho_dh * sigma_d
    C = (dDd**2) * (sigma_d**2) - (sigma_D_alvo**2)

    # Discriminante
    disc = B*B - 4.0*A*C
    if disc < 0:
        raise ValueError("Sem solução real para σ_h (verifique parâmetros).")
    # Raiz positiva
    sigma_h_req = (-B + math.sqrt(disc)) / (2.0*A)
    return sigma_h_req, D

# ------------------------------
# 2) Execução conforme preenchimento
# ------------------------------
print("=== Propagação padrão para D = sqrt(d^2 + h^2) ===")
print(f"d = {d:.3f} m,  h = {h:.3f} m")

if sigma_h is None and sigma_D_alvo is not None:
    # Resolver σ_h necessário
    sigma_h_req, D = resolver_sigma_h_para_sigma_D_alvo(d, h, sigma_d, sigma_D_alvo, rho_dh)
    print(f"D = {D:.3f} m")
    print(f"σ_D alvo = {sigma_D_alvo:.3f} m ({sigma_D_alvo*1000:.0f} mm)")
    print(f"⇒ σ_h necessário = {sigma_h_req:.3f} m  (~{sigma_h_req*1000:.0f} mm)")
else:
    # Calcular σ_D resultante com os σ informados
    if sigma_h is None:
        raise ValueError("Defina sigma_h ou sigma_D_alvo.")
    sigma_D, D = sigma_D_por_propagacao(d, h, sigma_d, sigma_h, rho_dh)
    print(f"D = {D:.3f} m")
    print(f"σ_d = {sigma_d:.6f} m,  σ_h = {sigma_h:.6f} m,  ρ = {rho_dh:.3f}")
    print(f"⇒ σ_D resultante = {sigma_D:.6f} m  (~{sigma_D*1000:.2f} mm)")
import numpy as np

# ============================================================
# AJUSTE DE REDE DE TRILATERAÇÃO
# Modelo Paramétrico NÃO Linear (Método de Newton / Jacobiana)
# ============================================================
# Conexão com os slides:
# - F(X_a) = L_b
# - J0 * X = K   (slide: "resolver J0·X = K, sendo X o vetor de correções")
# - Xa = X0 + X  (correções somadas ao chute)
# ============================================================


# ------------------------------------------------------------
# 1) DADOS FIXOS DO PROBLEMA  (corresponde ao "sejam Lb, Xa, X0..." do slide)
# ------------------------------------------------------------

# Pontos com coordenadas CONHECIDAS (injunções fixas)
P1 = np.array([100.0, 100.0])
P5 = np.array([300.0, 150.0])

# Observações L_b (distâncias medidas)  -> vetor m×1
obs = [
    (1, 2, 104.4226),
    (1, 3, 141.4264),
    (1, 4, 100.0186),
    (1, 6,  70.6993),
    (2, 3, 130.0119),
    (2, 4, 164.0010),
    (2, 5, 128.0688),
    (2, 6, 169.9940),
    (3, 4, 100.0009),
    (3, 5, 111.7834),
    (3, 6, 158.1090),
    (4, 5, 206.1490),
    (4, 6,  70.6874),
    (5, 6, 250.0094)
]

# Monta o vetor L_b com todas as distâncias observadas (m)
Lb = np.array([d for (_, _, d) in obs], dtype=float).reshape(-1, 1)

m = len(Lb)         # número de observações (equações)
sigma = 0.010       # desvio padrão de TODAS as observações (10 mm)

# Matriz de pesos P (slide: ela entra em N = J^T P J)
P = (1 / sigma**2) * np.eye(m)


# ------------------------------------------------------------
# 2) VETOR DE PARÂMETROS X (incógnitas) E "CHUTE INICIAL" X0
#    (conexão direta com o slide: X0 = aproximação inicial)
# ------------------------------------------------------------

# Vetor de parâmetros:
# X = [x2, y2, x3, y3, x4, y4, x6, y6]^T
x = np.array([
    200.0,  65.0,   # P2
    200.0, 200.0,   # P3
    100.0, 200.0,   # P4
     50.0, 150.0    # P6
], dtype=float)

# Guarda o número de parâmetros (n)
n = len(x)


# ------------------------------------------------------------
# 3) FUNÇÃO AUXILIAR PARA PEGAR COORDENADA DE UM PONTO
#    (usa X para pontos desconhecidos e valores fixos para P1 e P5)
# ------------------------------------------------------------

# Mapeamento de índice dentro do vetor x
mapa = {2: 0, 3: 2, 4: 4, 6: 6}  # ponto -> posição em x

def coord(pid, x):
    """
    Retorna as coordenadas (X,Y) do ponto pid.
    Se for ponto fixo (1 ou 5), usa valores conhecidos.
    Se for ponto desconhecido (2,3,4,6), lê no vetor x.
    """
    if pid == 1:
        return P1
    if pid == 5:
        return P5
    i = mapa[pid]
    return np.array([x[i], x[i+1]])


# ------------------------------------------------------------
# 4) DEFINIÇÃO DO MODELO F(X) = L_calc
#    (slide: F(X_a) = L_b -> aqui F(X) devolve as distâncias calculadas)
# ------------------------------------------------------------

def F(x):
    """
    Calcula o vetor F(x): distâncias calculadas entre todos os pares (i,j)
    F_k(x) = d_ij(x) = sqrt( (Xi - Xj)^2 + (Yi - Yj)^2 )
    """
    L_calc = []
    for (p1, p2, _) in obs:
        C1 = coord(p1, x)
        C2 = coord(p2, x)
        d  = np.sqrt(np.sum((C1 - C2)**2))
        L_calc.append(d)
    return np.array(L_calc).reshape(-1, 1)


# ------------------------------------------------------------
# 5) MATRIZ JACOBIANA J(X)  (slide: "seja J (matriz jacobiana de F)")
# ------------------------------------------------------------

def J_matrix(x):
    """
    Monta a matriz Jacobiana J(x),
    onde cada linha k corresponde à observação entre pontos (p1,p2):
      J_k,par = derivada de d_ij em relação ao parâmetro correspondente.
    Fórmula:
      d = sqrt(dx^2 + dy^2)
      ∂d/∂x_i =  dx/d
      ∂d/∂y_i =  dy/d
      ∂d/∂x_j = -dx/d
      ∂d/∂y_j = -dy/d
    """
    J = np.zeros((m, n), dtype=float)

    for k, (p1, p2, _) in enumerate(obs):

        C1 = coord(p1, x)
        C2 = coord(p2, x)
        dx = C1[0] - C2[0]
        dy = C1[1] - C2[1]
        d  = np.sqrt(dx*dx + dy*dy)

        # Se o ponto p1 é desconhecido, deriva em relação às suas coordenadas
        if p1 in mapa:
            j = mapa[p1]        # índice em x
            J[k, j]   =  dx / d # ∂d/∂x_p1
            J[k, j+1] =  dy / d # ∂d/∂y_p1

        # Se o ponto p2 é desconhecido, idem (com sinal trocado)
        if p2 in mapa:
            j = mapa[p2]
            J[k, j]   = -dx / d # ∂d/∂x_p2
            J[k, j+1] = -dy / d # ∂d/∂y_p2

    return J


# ------------------------------------------------------------
# 6) ITERAÇÃO DO MÉTODO DE NEWTON (slides 2, 3 e 4)
# ------------------------------------------------------------
# Slide: F(x0) + J0·X = Lb  =>  J0·X = Lb - F(x0) = K
# Aqui:
#   - x  = X0 (aproximação corrente)
#   - dx = X  (vetor de correções)
#   - K  = Lb - F(x)
# E resolvemos J0^T P J0 · dx = J0^T P K  (MMQ aplicado à equação J0·dx = K)
# ------------------------------------------------------------

tol = 1e-6
max_iter = 20

for it in range(1, max_iter + 1):

    # 1) Calcula F(x) na aproximação corrente (F(x0) no slide)
    F_x = F(x)

    # 2) Jacobiana J(x) = J0 (slide)
    J = J_matrix(x)

    # 3) Termo K = Lb - F(x0) (slide: "=> J0·X = Lb - F(x0) = K")
    K = Lb - F_x

    # 4) Sistema de MMQ para resolver J0·dx = K:
    #    Multiplicando por J0^T P dos dois lados: (J0^T P J0) dx = J0^T P K
    N = J.T @ P @ J       # N = J0^T P J0  (slide 5: N = J0^T P J0)
    U = J.T @ P @ K       # U = J0^T P K

    # 5) Resolve para as correções dx (slide: "resolver J0·X = K")
    dx = np.linalg.solve(N, U).flatten()

    # 6) Atualiza os parâmetros: Xa = X0 + X (slide 1: Xa = X0 + X)
    x_new = x + dx

    # 7) Informação de iteração
    print(f"\nIteração {it}:")
    print(f"  Correção máxima |dx| = {np.max(np.abs(dx)):.6e} m")
    print(f"  Parâmetros provisórios x = {x_new}")

    # 8) Critério de parada (slide 4: "critério de parada pode ser: max |X| < valor")
    if np.max(np.abs(dx)) < tol:
        x = x_new
        print("\nConvergência atingida (máxima correção abaixo do limite).")
        break

    # Se não convergiu, atualiza x0 := x_new e repete
    x = x_new


# ------------------------------------------------------------
# 7) RESULTADOS FINAIS DOS PARÂMETROS AJUSTADOS (Xa)
# ------------------------------------------------------------

print("\nPARÂMETROS AJUSTADOS (coordenadas dos pontos desconhecidos):")
for nome, (i, j) in zip(
    ["P2", "P3", "P4", "P6"],
    [(0, 1), (2, 3), (4, 5), (6, 7)]
):
    print(f"  {nome}: X = {x[i]:.4f} m  |  Y = {x[j]:.4f} m")


# ------------------------------------------------------------
# 8) CÁLCULO DA MATRIZ COVARIÂNCIA Σx  (slide 5)
# ------------------------------------------------------------
#   Σx = σ0² · N^{-1},  com σ0² = (v^T P v) / (m - n)
# onde:
#   - v = resíduos no espaço das observações = F(Xa) - Lb
# ------------------------------------------------------------

# Recalcula F(Xa)
F_a = F(x)

# Resíduos: observado - calculado OU calculado - observado (mantém consistente)
v = F_a - Lb   # aqui: v = F(Xa) - Lb

# Variância unitária posterior σ0²
dof = m - n         # graus de liberdade = nº obs - nº parâmetros
sigma0_sq = (v.T @ P @ v) / dof
sigma0_sq = sigma0_sq.item()

# Matriz covariância dos parâmetros
N_final = N                  # N da última iteração
Sigma_x = np.linalg.inv(N_final)

print("\nVARIÂNCIA UNITÁRIA POSTERIOR:")
print(f"  sigma0² = {sigma0_sq:.3e}  ->  sigma0 = {np.sqrt(sigma0_sq):.3e} m")

print("\nMATRIZ COVARIÂNCIA x (parâmetros):")
print(Sigma_x)
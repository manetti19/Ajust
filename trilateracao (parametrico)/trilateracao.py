import numpy as np

# -----------------------------
# 1) Defina pontos FIXOS
# -----------------------------
fixed = {
    1: np.array([100.0, 100.0]),   # P1
    5: np.array([300.0, 150.0])    # P5
}

# -----------------------------
# 2) Aproximações iniciais
# -----------------------------
# Vetor de parâmetros na ordem: [X2,Y2, X3,Y3, X4,Y4, X6,Y6]
x = np.array([
    200.0,  65.0,   # P2
    200.0, 200.0,   # P3
    100.0, 200.0,   # P4
     50.0, 150.0    # P6
], dtype=float)

# id -> índices dentro de x
unknown_ids = [2, 3, 4, 6]                      # ids desconhecidos
offset = {pid: 2*i for i, pid in enumerate(unknown_ids)}  # inicio (X) de cada ponto em x

# -----------------------------
# 3) Observações de distância (De, Para, Distância em metros)
#    (copiadas da sua tabela da tela)
# -----------------------------
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
    (5, 6, 250.0094),
]

# -----------------------------
# 4) Pesos e controle de iteração
# -----------------------------
sigma_dist = 0.010                      # 10 mm
P = (1.0 / sigma_dist**2) * np.eye(len(obs))  # todos com mesmo peso
tol = 1e-8                               # tolerância para correção de parâmetro
max_iter = 30                            # limite de iterações

# -----------------------------
# 5) Funções auxiliares
# -----------------------------
def coord_of(pid: int, xvec: np.ndarray) -> np.ndarray:
    """Retorna coordenada (X,Y) do ponto pid; usa 'fixed' se conhecido ou extrai de x se desconhecido."""
    if pid in fixed:
        return fixed[pid]
    j = offset[pid]
    return np.array([xvec[j], xvec[j+1]])

def add_partials_row(A, irow, p_from, p_to, C_from, C_to, d):
    """Preenche derivadas da i-ésima observação (linha de A) para pontos desconhecidos."""
    if d == 0.0:
        raise ValueError("D=0")
    dx = C_from[0] - C_to[0]
    dy = C_from[1] - C_to[1]
    # contribuições: + (dL/dX_from, dL/dY_from) e - para 'to'
    if p_from in offset:
        j = offset[p_from]
        A[irow, j]   =  dx / d
        A[irow, j+1] =  dy / d
    if p_to in offset:
        k = offset[p_to]
        A[irow, k]   = -dx / d
        A[irow, k+1] = -dy / d

# -----------------------------
# 6) Loop Gauss-Newton
# -----------------------------
n = len(obs)         # número de observações
u = len(x)           # número de parâmetros (8)
for it in range(1, max_iter+1):
    A  = np.zeros((n, u))
    l0 = np.zeros((n, 1))   # distâncias calculadas (aproximação)
    L  = np.zeros((n, 1))   # distâncias observadas

    # monta A, l0 e L
    for i, (p_from, p_to, dij) in enumerate(obs):
        C_from = coord_of(p_from, x)
        C_to   = coord_of(p_to,   x)
        diff   = C_from - C_to
        dcalc  = float(np.hypot(diff[0], diff[1]))   # distância calculada
        l0[i, 0] = dcalc
        L[i, 0]  = dij
        add_partials_row(A, i, p_from, p_to, C_from, C_to, dcalc)

    # modelo linearizado: v = A·dx - (L - l0)
    w = L - l0                         # discrepâncias obs - calc
    N = A.T @ P @ A                    # normais
    nvec = A.T @ P @ w                 # termo independente
    dx = np.linalg.solve(N, nvec)      # correção dos parâmetros
    x  = x + dx.flatten()              # atualiza estimativas

    # critério de parada
    if np.max(np.abs(dx)) < tol:
        print(f"Convergiu na iteração {it}.")
        break
else:
    print("Aviso: atingiu o máximo de iterações sem convergir.")

# -----------------------------
# 7) Pós-ajuste: resíduos, sigma0² e saída
# -----------------------------
# Recalcula A, w com parâmetros finais para obter v
A  = np.zeros((n, u))
l0 = np.zeros((n, 1))
L  = np.zeros((n, 1))
for i, (p_from, p_to, dij) in enumerate(obs):
    C_from = coord_of(p_from, x)
    C_to   = coord_of(p_to,   x)
    diff   = C_from - C_to
    dcalc  = float(np.hypot(diff[0], diff[1]))
    l0[i, 0] = dcalc
    L[i, 0]  = dij
    add_partials_row(A, i, p_from, p_to, C_from, C_to, dcalc)

w = L - l0
# Obtém v pela equação normal (v = A·dx - w). Com dx final ≈ 0, v ≈ -w + projeção no espaço de A.
# Compute v com fórmula padrão: v = A·(N^{-1}A^T P w) - w
N = A.T @ P @ A
dx_hat = np.linalg.solve(N, A.T @ P @ w)
v = A @ dx_hat - w

dof = n - u
sigma0_sq = float(v.T @ P @ v / dof)

# -----------------------------
# 8) Relatório
# -----------------------------
def pr_point(pid, label):
    j = offset[pid]
    return f"{label}: X = {x[j]:.4f}  Y = {x[j+1]:.4f}"

print("\n--- Coordenadas ajustadas (m) ---")
print(pr_point(2, "P2"))
print(pr_point(3, "P3"))
print(pr_point(4, "P4"))
print(pr_point(6, "P6"))

print(f"\n(variância unitária posterior): {sigma0_sq:.3e}  ( = {np.sqrt(sigma0_sq):.3e} m)")
print("\nResíduos (m) por observação (De-Para : v):")
for i, (p_from, p_to, _) in enumerate(obs):
    print(f"{p_from}-{p_to}: {float(v[i,0]): .4f}")

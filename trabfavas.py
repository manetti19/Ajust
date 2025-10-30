import numpy as np

# ===========================================================
# Ajuste de circunferência pelo Modelo Combinado (Gauss-Helmert)
# ===========================================================
# Função: F = (Xi - Xc)^2 + (Yi - Yc)^2 - r^2 = 0
# Parâmetros: Xc, Yc, r
# Observações (ajustáveis): Xi, Yi (i=1..4)
# Pesos: P = diag(1/sigma^2) na ordem [X1,Y1,X2,Y2,X3,Y3,X4,Y4]
# Vetor inicial: [Xc0, Yc0, r0] = [100, 120, 70]
#
# O script implementa:
#  - modo="apostila": A e B fixos em (X0, Lb); em cada iteração atualiza W,
#                     resolve X, K e V; atualiza Xa e La; repete.
#  - modo="kkt":      monta e resolve o sistema KKT relinearizando A,B a cada passo.
#
# Impressões: mostra A, B, W da 1ª iteração (batendo com as lâminas),
#             e o histórico das 4 iterações (Xa, La).
# ===========================================================

# ---------- Dados de entrada (iguais às figuras) ----------
X_obs = np.array([140.0, 165.0, 165.0, 140.0], dtype=float)  # X1..X4
Y_obs = np.array([ 60.0, 100.0, 150.0, 180.0], dtype=float)  # Y1..Y4

sig2_x = np.array([2.0, 4.0, 2.0, 4.0], dtype=float)  # var(X)
sig2_y = np.array([2.0, 4.0, 2.0, 4.0], dtype=float)  # var(Y)

# Empilha Lb = [X1,Y1,X2,Y2,X3,Y3,X4,Y4]^T
Lb = np.empty(8, dtype=float)
Lb[0::2] = X_obs
Lb[1::2] = Y_obs

# P = diag(1/sigma^2) com a mesma ordem de Lb
sig2_stack = np.empty(8, dtype=float)
sig2_stack[0::2] = sig2_x
sig2_stack[1::2] = sig2_y
P = np.diag(1.0 / sig2_stack)
Pinv = np.diag(sig2_stack)  # P^{-1} = diag(σ^2)

# Estado inicial
Xa = np.array([100.0, 120.0, 70.0], dtype=float)  # [Xc, Yc, r]
La = Lb.copy()                                    # observações ajustadas

# ---------- Funções auxiliares ----------
def F_eval(Xa, La):
    """ Vetor F (m=4): (Xi - Xc)^2 + (Yi - Yc)^2 - r^2 """
    Xc, Yc, r = Xa
    F = np.zeros(4, dtype=float)
    for i in range(4):
        Xi = La[2*i]; Yi = La[2*i+1]
        F[i] = (Xi - Xc)**2 + (Yi - Yc)**2 - r**2
    return F

def AB_eval(Xa, La):
    """ Matrizes A (4x3) e B (4x8) avaliadas no par (Xa, La). """
    Xc, Yc, r = Xa
    A = np.zeros((4, 3), dtype=float)
    B = np.zeros((4, 8), dtype=float)
    for i in range(4):
        Xi = La[2*i]; Yi = La[2*i+1]
        # Derivadas em relação aos parâmetros (Xc, Yc, r)
        A[i, 0] = -2.0 * (Xi - Xc)
        A[i, 1] = -2.0 * (Yi - Yc)
        A[i, 2] = -2.0 * r
        # Derivadas em relação às observações (Xi, Yi do ponto i)
        B[i, 2*i    ] = +2.0 * (Xi - Xc)
        B[i, 2*i + 1] = +2.0 * (Yi - Yc)
    return A, B

def passo_apostila(Xa, La, Lb, P, Pinv):
    """
    Um passo do método 'apostila' porém com B_i (e A_i) atualizados,
    de forma consistente com as lâminas:

        W_i = B_i (Lb - La^{i-1}) + F(Xa^{i-1}, La^{i-1})
        M_i = B_i P^{-1} B_i^T
        X = -(A_i^T M_i^{-1} A_i)^{-1} A_i^T M_i^{-1} W_i
        K = -M_i^{-1}(A_i X + W_i)
        V = P^{-1} B_i^T K
        La = Lb + V
        Xa = Xa + X
    """
    # 1) Atualiza derivadas na iteração i
    A_i, B_i = AB_eval(Xa, La)

    # 2) Define W_i com B_i ATUAL
    W_i = B_i @ (Lb - La) + F_eval(Xa, La)

    # 3) Monta M_i e resolve correções
    M_i = B_i @ Pinv @ B_i.T
    Y_i = np.linalg.solve(M_i, A_i)          # M_i^{-1} A_i
    g_i = np.linalg.solve(M_i, W_i)          # M_i^{-1} W_i
    N_i = A_i.T @ Y_i                         # A_i^T M_i^{-1} A_i
    Xcorr = -np.linalg.solve(N_i, A_i.T @ g_i)

    # 4) Lagrangeanos e correções das observações
    K_i = -np.linalg.solve(M_i, A_i @ Xcorr + W_i)
    V_i = Pinv @ (B_i.T @ K_i)

    # 5) Atualizações (La = Lb + V, conforme lâmina)
    Xa_new = Xa + Xcorr
    La_new = Lb + V_i

    return Xa_new, La_new, Xcorr, V_i, W_i

# ---------- Preparação do modo "apostila" ----------
# Derivadas na linearização inicial (X0, Lb) — batem com as matrizes da figura
A0, B0 = AB_eval(np.array([100.0, 120.0, 70.0]), Lb)
M0 = B0 @ Pinv @ B0.T

# Mostra A, B, W da 1ª iteração (iguais às lâminas)
W0 = F_eval(np.array([100.0,120.0,70.0]), Lb)
print("A (X0,Lb) =\n", A0)
print("\nB (X0,Lb) =\n", B0)
print("\nM =\n", M0)
print("\nW = F(X0,Lb) =\n", W0)

# ---------- Rodar 4 iterações com B atualizado em W ----------
Xa_ap = Xa.copy()
La_ap = La.copy()
hist_ap = []
for k in range(1, 5):
    Xa_ap, La_ap, Xcorr, V, W = passo_apostila(Xa_ap, La_ap, Lb, P, Pinv)
    hist_ap.append((k, Xa_ap.copy(), La_ap.copy()))

print("\n=== Histórico (B atualizado em W) ===")
for k, Xa_k, La_k in hist_ap:
    print(f"Iter {k}: Xa = {Xa_k},  La = {La_k}")

import numpy as np

# -----------------------------------
# 1️⃣ Dados do problema
# -----------------------------------
# Pontos observados (X, Y)
pontos = np.array([
    [140, 60],
    [165, 100],
    [165, 150],
    [140, 180]
], dtype=float)


# Variâncias das observações
sigma_x2 = np.array([2, 4, 2, 4], dtype=float)  # σx^2
sigma_y2 = np.array([2, 4, 2, 4], dtype=float)  # σy^2

# Vetor de observações brutas Lb (X1,Y1,X2,Y2,...)
Lb = np.zeros(8)
Lb[0::2] = pontos[:, 0]  # X1, X2, ...
Lb[1::2] = pontos[:, 1]  # Y1, Y2, ...
La = Lb.copy()
Vfinal = np.zeros(8, dtype=float)

wX = 1/sigma_x2
wY = 1/sigma_y2
w_inter = np.zeros(8, dtype=float)
w_inter[0::2] = wX      # X1, X2, X3, X4
w_inter[1::2] = wY      # Y1, Y2, Y3, Y4
P = np.diag(w_inter)
Pinv = np.diag(1/np.diag(P))


# Critérios de convergência
tol = 1e-6
max_iter = 50
erro = 1e10
iteracao = 0

# Chute inicial para os parâmetros [Xc, Yc, r]
Xc0 = 100.0
Yc0 = 120.0
r0 = 70.0
X_params = np.array([Xc0, Yc0, r0], dtype=float)
delta_Xfinal = np.zeros(3)

# -----------------------------------
# 2️⃣ Função de condição da circunferência
# -----------------------------------
def F_n(Xc, Yc, r, Xn, Yn):
    return (Xn - Xc)**2 + (Yn - Yc)**2 - r**2


# -----------------------------------
# 3️⃣ Iteração do ajuste combinado
# -----------------------------------
while erro > tol and iteracao < max_iter:
    
    A = np.zeros((4, 3), dtype=float)   # 4 equações, 3 parâmetros
    B = np.zeros((4, 8), dtype=float)   # 4 equações, 8 observações
    W = np.zeros(4, dtype=float)        # vetor de fechamento
    F = np.zeros(4, dtype=float)
    Xc, Yc, r = X_params


    # Construir A, B, W
    for i in range(4):
        Xn = La[2*i]
        Yn = La[2*i + 1]

        # Derivadas em relação aos parâmetros (A)
        A[i, 0] = -2*(Xn - Xc)   # dF/dXc
        A[i, 1] = -2*(Yn - Yc)   # dF/dYc
        A[i, 2] = -2*r            # dF/dr
        
        # Derivadas em relação às observações (B)
        B[i, 2*i] = 2*(Xn - Xc)   # dF/dXn
        B[i, 2*i+1] = 2*(Yn - Yc) # dF/dYn
        F[i] = F_n(Xc, Yc, r, Xn, Yn)

    # termo de fechamento correto
    W = B @ (Lb - La) + F
    

    PinvBT = Pinv @ B.T
    M = B @ PinvBT
    '''M = np.array([
    [41600.,     0.,     0.,     0.],
    [    0., 74000.,     0.,     0.],
    [    0.,     0., 41000.,     0.],
    [    0.,     0.,     0., 83200.]], dtype=float)'''


    #print(M)
    MinvA = np.linalg.solve(M, A)
    MinvW = np.linalg.solve(M, W)

    N = A.T @ MinvA
    u = A.T @ MinvW
    try:
        delta_X = -np.linalg.solve(N, u)
    except np.linalg.LinAlgError:
        delta_X = -np.linalg.lstsq(N, u, rcond=None)[0]

    delta_Xfinal += delta_X
    X_params += delta_X
    
    rhsK = A @ X_params + W
    K = -np.linalg.solve(M, rhsK)

    V = Pinv @ (B.T @ K)
    
    Vfinal += V
    La = Lb + V
    

    # Erro para critério de convergência
    erro = np.linalg.norm(delta_X)
    iteracao += 1


# -----------------------------------
# 8️⃣ Resultados finais
# -----------------------------------
Xc, Yc, r = X_params
print(f"Convergiu em {iteracao} iterações")
print(f"delta_X final: \n {delta_Xfinal}")
print(f"Centro da circunferência: Xc = {Xc:.4f}, Yc = {Yc:.4f}")
print(f"Raio ajustado: r = {r:.4f}")
print(f"Observações ajustadas (La):\n{La}")
print(f"Resíduos das observações (V):\n{Vfinal}")

import numpy as np

'''
# 2 caso com Xc0, Yc0, r0 = media dos pontos 

# -----------------------------------
# 1️⃣ Dados do problema
# -----------------------------------
# Pontos observados (X, Y)
pontos = np.array([
    [140, 60],
    [165, 100],
    [165, 150],
    [140, 180]
])

# Variâncias das observações
sigma_x2 = np.array([2, 4, 2, 4])  # σx^2
sigma_y2 = np.array([2, 4, 2, 4])  # σy^2

# Vetor de observações brutas Lb (X1,Y1,X2,Y2,...)
Lb = np.zeros(8)
Lb[0::2] = pontos[:, 0]  # X1, X2, ...
Lb[1::2] = pontos[:, 1]  # Y1, Y2, ...
La = Lb.copy()


# Matriz de pesos P (8x8 diagonal)
P = np.diag(np.concatenate([1/sigma_x2, 1/sigma_y2]))

# Chute inicial para os parâmetros [Xc, Yc, r]
Xc0 = np.mean(pontos[:, 0])
Yc0 = np.mean(pontos[:, 1])
r0 = np.mean(np.sqrt((pontos[:,0]-Xc0)**2 + (pontos[:,1]-Yc0)**2))
X_params = np.array([Xc0, Yc0, r0], dtype=float)

# Critérios de convergência
tol = 1e-6
max_iter = 50
erro = 1e10
iteracao = 0

# -----------------------------------
# 2️⃣ Função de condição da circunferência
# -----------------------------------
def F_n(Xc, Yc, r, Xn, Yn):
    return (Xn - Xc)**2 + (Yn - Yc)**2 - r**2



# -----------------------------------
# 3️⃣ Iteração do ajuste combinado
# -----------------------------------
while erro > tol and iteracao < max_iter:
    
    A = np.zeros((4, 3))   # 4 equações, 3 parâmetros
    B = np.zeros((4, 8))   # 4 equações, 8 observações
    W = np.zeros(4)        # vetor de fechamento
    
    # Construir A, B, W
    for i in range(4):
        Xn, Yn = pontos[i]
        Xc, Yc, r = X_params
        
        # Função de fechamento W
        W[i] = F_n(Xc, Yc, r, Xn, Yn)
        
        # Derivadas em relação aos parâmetros (A)
        A[i, 0] = -2*(Xn - Xc)   # dF/dXc
        A[i, 1] = -2*(Yn - Yc)   # dF/dYc
        A[i, 2] = -2*r            # dF/dr
        
        # Derivadas em relação às observações (B)
        B[i, 2*i] = 2*(Xn - Xc)   # dF/dXn
        B[i, 2*i+1] = 2*(Yn - Yc) # dF/dYn
    
    # -----------------------------------
    # 4️⃣ Matriz M e multiplicadores de Lagrange
    # -----------------------------------
    M = B @ np.linalg.inv(P) @ B.T
    # Resolver multiplicadores: K = -M^{-1} (A X + W)
    K = -np.linalg.solve(M, A @ X_params + W)
    
    # -----------------------------------
    # 5️⃣ Resíduos das observações
    # -----------------------------------
    V = np.linalg.inv(P) @ B.T @ K
    
    # -----------------------------------
    # 6️⃣ Atualização das observações ajustadas
    # -----------------------------------
    La += V  # observações ajustadas

    # -----------------------------------
    # 7️⃣ Atualização dos parâmetros
    # -----------------------------------
    delta_X = -np.linalg.solve(A.T @ np.linalg.inv(M) @ A, A.T @ np.linalg.inv(M) @ W)
    X_params += delta_X
    
    # Erro para critério de convergência
    erro = np.linalg.norm(delta_X)
    iteracao += 1



print(M)
print(K)
print(V)
# -----------------------------------
# 8️⃣ Resultados finais
# -----------------------------------
Xc, Yc, r = X_params
print(f"Convergiu em {iteracao} iterações")
print(f"Centro da circunferência: Xc = {Xc:.4f}, Yc = {Yc:.4f}")
print(f"Raio ajustado: r = {r:.4f}")
print(f"Observações ajustadas (La):\n{La}")
print(f"Resíduos das observações (V):\n{V}")


'''








# 2 caso com Xc0 = 100, Yc0 = 120, r0 = 70

# -----------------------------------
# 1️⃣ Dados do problema
# -----------------------------------
# Pontos observados (X, Y)
pontos = np.array([
    [140, 60],
    [165, 100],
    [165, 150],
    [140, 180]
])


# Variâncias das observações
sigma_x2 = np.array([2, 4, 2, 4])  # σx^2
sigma_y2 = np.array([2, 4, 2, 4])  # σy^2

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


A = np.zeros((4, 3))   # 4 equações, 3 parâmetros
B = np.zeros((4, 8))   # 4 equações, 8 observações
W = np.zeros(4)        # vetor de fechamento

# -----------------------------------
# 3️⃣ Iteração do ajuste combinado
# -----------------------------------
while erro > tol and iteracao < max_iter:
    
    # Construir A, B, W
    for i in range(4):
        Xn = La[2*i]
        Yn = La[2*i + 1]
        Xc, Yc, r = X_params
   
        # Função de fechamento W
        W[i] = F_n(Xc, Yc, r, Xn, Yn)

        # Derivadas em relação aos parâmetros (A)
        A[i, 0] = -2*(Xn - Xc)   # dF/dXc
        A[i, 1] = -2*(Yn - Yc)   # dF/dYc
        A[i, 2] = -2*r            # dF/dr
        
        # Derivadas em relação às observações (B)
        B[i, 2*i] = 2*(Xn - Xc)   # dF/dXn
        B[i, 2*i+1] = 2*(Yn - Yc) # dF/dYn
    
    PinvBT = np.linalg.solve(P, B.T)
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
    delta_X = -np.linalg.solve(N, u)

    delta_Xfinal += delta_X
    X_params += delta_X


    K = -np.linalg.solve(M, A @ delta_X + W)

    V = np.linalg.solve(P, B.T @ K)
    
    Vfinal += V
    La += V
    
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

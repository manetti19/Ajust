import numpy as np  # importa o numpy para contas matriciais

def mmq(A, L, P=None):
    """
    Resolve A x = L + v por Mínimos Quadrados (pesos opcionais).
    Retorna dict com x_hat, v, L_hat, N, U, sigma0_sq, Qxx, Cxx, std_x.
    """
    A = np.asarray(A, dtype=float)                 # garante array float
    L = np.asarray(L, dtype=float).reshape(-1)     # garante vetor 1-D
    n, u = A.shape                                 # n = nº obs, u = nº incógnitas

    # Matriz de pesos: identidade se não fornecida
    if P is None:
        P = np.eye(n, dtype=float)                 # pesos iguais (P = I)
    else:
        P = np.asarray(P, dtype=float)

    # Monta normais: N = A^T P A ; U = A^T P L
    N = A.T @ P @ A                                # matriz normal
    U = A.T @ P @ L                                # termo independente

    # Resolve N x = U para obter x_hat
    x_hat = np.linalg.solve(N, U)                  # solução de variância mínima

    # Resíduos: v = A x_hat - L  (modelo: A x = L + v)
    v = A @ x_hat - L                              # vetor de resíduos

    # Observações ajustadas
    L_hat = A @ x_hat                              # L chapéu

    # Variância a posteriori do unit weight: sigma0^2 = v^T P v / (n - u)
    dof = n - u                                    # graus de liberdade
    sigma0_sq = float(v.T @ P @ v) / dof           # variância escalar a posteriori

    # Precisão dos parâmetros: Qxx = N^{-1} ; Cxx = sigma0^2 * Qxx
    Qxx = np.linalg.inv(N)                         # matriz de cofatores dos parâmetros
    Cxx = sigma0_sq * Qxx                          # covariância dos parâmetros
    std_x = np.sqrt(np.diag(Cxx))                  # desvios-padrão de x_hat

    # Empacota tudo num dicionário
    return {
        "x_hat": x_hat,
        "v": v,
        "L_hat": L_hat,
        "N": N,
        "U": U,
        "sigma0_sq": sigma0_sq,
        "Qxx": Qxx,
        "Cxx": Cxx,
        "std_x": std_x,
        "dof": dof
    }

# =========================
# EXEMPLO – sua 5ª questão
# Modelo A x = L  (x = [x1, x2, x3]^T)
# x1              = -3.0122   ->  [ 1,  0,  0] x = -3.0122
# -x1 + x2        =  1.0076   ->  [-1,  1,  0] x =  1.0076
#      -x2 + x3   =  0.9958   ->  [ 0, -1,  1] x =  0.9958
#           -x3   =  1.0050   ->  [ 0,  0, -1] x =  1.0050
#          -x2    =  2.0058   ->  [ 0, -1,  0] x =  2.0058
# -x1       + x3  =  2.0054   ->  [-1,  0,  1] x =  2.0054

A = np.array([
    [ 1,  0,  0],
    [-1,  1,  0],
    [ 0, -1,  1],
    [ 0,  0, -1],
    [ 0, -1,  0],
    [-1,  0,  1]
], dtype=float)

L = np.array([-3.0122, 1.0076, 0.9958, 1.0050, 2.0058, 2.0054], dtype=float)

# Chamada com pesos iguais (P = I)
res = mmq(A, L)

# Impressão bonitinha dos resultados principais
np.set_printoptions(precision=6, suppress=True)
print("x_hat (parâmetros ajustados)  =", res["x_hat"])
print("std_x (desvios-padrão de x)   =", res["std_x"])
print("v (resíduos)                  =", res["v"])
print("L_hat (obs. ajustadas)        =", res["L_hat"])
print("sigma0^2                      =", res["sigma0_sq"])
print("N = A^T P A                   =\n", res["N"])
print("U = A^T P L                   =\n", res["U"])
print("Qxx = N^{-1}                  =\n", res["Qxx"])
print("Cxx = sigma0^2 * Qxx          =\n", res["Cxx"])
print("graus de liberdade            =", res["dof"])
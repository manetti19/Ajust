import numpy as np
import time         # cronometrar com perf_counter
t0 = time.perf_counter()       # inicia cronômetro da iteração

def F(vec):
    """
    Vetor de resíduos F(x) = [f1, f2, f3]^T.
    Cada fi já está 'levado para a esquerda' (RHS subtraído).
    """
    x, y = float(vec[0]), float(vec[1])       # desempacota as incógnitas
    f1 = 5*x - 7*y + 14
    f2 = 3*x + y - 8
    f3 = x - 9*y + 23
    f4 = 3*x - y - 3
    f5 = 6*x - y - 6

    return np.array([f1, f2, f3, f4, f5], dtype=float)

def J(vec):
    """
    Jacobiana J(x) (m x n): derivadas parciais de cada fi
    em relação a (x, y), na mesma ordem de F().
    """
    x, y = float(vec[0]), float(vec[1])       # desempacota as incógnitas
    df1dx, df1dy = 5, -7       # gradiente de f1
    df2dx, df2dy = 3, 1        # gradiente de f2
    df3dx, df3dy = 1, -9       # gradiente de f3
    df4dx, df4dy = 3, -1       # gradiente de f4
    df5dx, df5dy = 6, -1       # gradiente de f5
    return np.array([[df1dx, df1dy],
                     [df2dx, df2dy],
                     [df3dx, df3dy],
                     [df4dx, df4dy],
                     [df5dx, df5dy]], dtype=float)

def newton_system(x, tol=1e-12, maxit=50, verbose=True):
    x = np.array(x0, float)
    for k in range(1, maxit+1):              # laço de iterações
        r  = F(x)                                        # residual no ponto atual
        Jx = J(x)                                        # Jacobiana no ponto atual
        N  = Jx.T @ Jx                                   # normais (2x2)
        U  = - Jx.T @ r                                  # termo independente

        L  = np.linalg.cholesky(N)                       # N = L L^T (L inferior)
        y  = np.linalg.solve(L, U)                       # L y = U   (direta)
        d  = np.linalg.solve(L.T, y)                     # L^T d = y (retro)
        x_new = x + d                                    # atualização
        
        # Métricas de convergência
        step_norm = np.linalg.norm(d, ord=np.inf)  # norma do passo
        res_norm  = np.linalg.norm(F(x_new))        # norma do novo residual

        # (opcional) imprime a tabela de iteração
        if verbose:
            print(f"it={k:2d}  x={x_new[0]: .10f}  y={x_new[1]: .10f}  "
                  f"||Δ||_∞={step_norm:.3e}  ||F||_2={res_norm:.3e}")

        # Critério de parada: passo pequeno OU residual pequeno
        if step_norm < tol or res_norm < tol:
            return x_new, k, res_norm

        x = x_new                               # avança para a próxima iteração

    # Se chegar aqui, não convergiu dentro de maxit (retorna o último x)
    return x, maxit, np.linalg.norm(F(x))

x0 = [2.5, 0.4]                             # x0=2,5 ; y0=0,4
sol, it, nF = newton_system(x0, tol=1e-12, maxit=50, verbose=True)
print("\nSolução aproximada:")
print(f"x ≈ {sol[0]:.12f}")
print(f"y ≈ {sol[1]:.12f}")
print(f"||F(x)||₂ ≈ {nF:.6e}")             # norma do residual final
print(f"F(x) = {F(sol)}")                  # mostra os 3 resíduos nas equações

    
dt = time.perf_counter() - t0            # tempo total em segundos
print(f"Tempo total: {dt:.6f} s ({dt*1000:.3f} ms)")

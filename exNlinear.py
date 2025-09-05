import numpy as np  # biblioteca para álgebra linear
import time         # cronometrar com perf_counter
t0 = time.perf_counter()       # inicia cronômetro da iteração

# =========================
# DEFINIÇÃO DO PROBLEMA
# =========================
def F(vec):
    """
    Vetor de resíduos F(x) = [f1, f2, f3]^T.
    Cada fi já está 'levado para a esquerda' (RHS subtraído).
    """
    x, y = float(vec[0]), float(vec[1])       # desempacota as incógnitas
    f1 = x**2 + 3*x*y - y**2 - 7.0           # x^2 + 3xy - y^2 = 7,0  ->  f1 = ... - 7.0
    f2 = 7*x**3 - 3*y**2 - 55.2              # 7x^3 - 3y^2 = 55,2     ->  f2 = ... - 55.2
    f3 = 2*x - 6*x*y + 3*y**2 + 1.2          # 2x - 6xy + 3y^2 = -1,2 ->  f3 = ... - (-1.2) = ... + 1.2
    return np.array([f1, f2, f3], dtype=float)

def J(vec):
    """
    Jacobiana J(x) (m x n): derivadas parciais de cada fi
    em relação a (x, y), na mesma ordem de F().
    """
    x, y = float(vec[0]), float(vec[1])       # desempacota as incógnitas
    df1dx, df1dy = 2*x + 3*y, 3*x - 2*y       # gradiente de f1
    df2dx, df2dy = 21*x**2,   -6*y            # gradiente de f2
    df3dx, df3dy = 2 - 6*y,   -6*x + 6*y      # gradiente de f3
    return np.array([[df1dx, df1dy],
                     [df2dx, df2dy],
                     [df3dx, df3dy]], dtype=float)

# ===========================================
# SOLVER: NEWTON / GAUSS-NEWTON AUTOMÁTICO
# ===========================================
def newton_system(x0, tol=1e-12, maxit=50, verbose=True):
    """
    Resolve F(x)=0.
    - Se m == n: usa Newton clássico:  J Δ = -F
    - Se m  > n: usa Gauss-Newton:    (J^T J) Δ = -J^T F
    """
    x = np.array(x0, dtype=float)            # chute inicial como vetor numpy
    for k in range(1, maxit+1):              # laço de iterações
        r = F(x)                              # residual F(x_k)
        Jx = J(x)                             # Jacobiana J(x_k)
        m, n = Jx.shape                       # dimensões (m eqs, n incognitas)

        # Decide o sistema linear da etapa (Newton clássico ou Gauss-Newton)
        if m == n:
            A = Jx                            # matriz do sistema linear
            b = -r                            # lado direito
        else:
            A = Jx.T @ Jx                     # normal equations (Gauss-Newton)
            b = -Jx.T @ r                     # termo independente

        # Resolve a etapa Δ (com fallback para lstsq se A for singular)
        try:
            delta = np.linalg.solve(A, b)     # resolve A Δ = b
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(A, b, rcond=None)  # pseudo-solução

        x_new = x + delta                     # atualização: x_{k+1} = x_k + Δ

        # Métricas de convergência
        step_norm = np.linalg.norm(delta, ord=np.inf)  # norma do passo
        res_norm  = np.linalg.norm(F(x_new))           # norma do novo residual

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

# =========================
# EXECUÇÃO COM O SEU CHUTE
# =========================
if __name__ == "__main__":
    x0 = [2.5, 0.4]                             # x0=2,5 ; y0=0,4
    sol, it, nF = newton_system(x0, tol=1e-12, maxit=100, verbose=True)
    print("\nSolução aproximada:")
    print(f"x ≈ {sol[0]:.12f}")
    print(f"y ≈ {sol[1]:.12f}")
    print(f"||F(x)||₂ ≈ {nF:.6e}")             # norma do residual final (m>n → nem sempre zera)
    print(f"F(x) = {F(sol)}")                  # mostra os 3 resíduos nas equações

    
dt = time.perf_counter() - t0            # tempo total em segundos
print(f"Tempo total: {dt:.6f} s ({dt*1000:.3f} ms)")

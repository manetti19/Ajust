import numpy as np

# =========================
# 1) ENTRADAS DO PROBLEMA
# =========================

# Estações conhecidas (fixas): nome -> (X, Y)
fixed = {
    "1": (100.0, 100.0),
    "5": (300.0, 150.0),
}

# Estações desconhecidas (chute inicial): nome -> (X0, Y0)
unknown0 = {
    "2": (200.0,  65.0),
    "3": (200.0, 200.0),
    "4": (100.0, 200.0),
    "6": ( 50.0, 150.0),
}

# Observações de distância: (de, para, d_obs [m], sigma_d [m])
sigma = 0.01
obs = [
    ("1","2",104.4226,sigma),
    ("1","3",141.4264,sigma),
    ("1","4",100.0186,sigma),
    ("1","6", 70.6993,sigma),
    ("2","3",130.0119,sigma),
    ("2","4",164.0010,sigma),
    ("2","5",128.0688,sigma),
    ("2","6",169.9940,sigma),
    ("3","4",100.0009,sigma),
    ("3","5",111.7834,sigma),
    ("3","6",158.1090,sigma),
    ("4","5",206.1490,sigma),
    ("4","6", 70.6874,sigma),
    ("5","6",250.0094,sigma),
]

# =========================
# 2) MONTAGEM DOS PARÂMETROS
# =========================
unk_names = list(unknown0.keys())
n = 2 * len(unk_names)                              # nº de incógnitas
param_index = {name:(2*i, 2*i+1) for i,name in enumerate(unk_names)}

X = np.zeros(n, dtype=float)                        # vetor de parâmetros
for name,(x0,y0) in unknown0.items():
    i,j = param_index[name]
    X[i], X[j] = x0, y0

P = np.diag([1.0/(s**2) for *_,s in obs])          # pesos (diagonal)

def get_xy(name, Xvec):
    if name in fixed: return fixed[name]
    i,j = param_index[name]
    return Xvec[i], Xvec[j]

# =========================
# 3) FUNÇÕES AUXILIARES
# =========================
def build_w_A(Xvec):
    """ w = d_obs - d_calc  ;  A = derivadas em relação às incógnitas (x,y dos pontos desconhecidos). """
    m = len(obs)
    w = np.zeros(m, dtype=float)
    A = np.zeros((m, n), dtype=float)

    for k,(a,b,d_obs,sig) in enumerate(obs):
        xa,ya = get_xy(a, Xvec)
        xb,yb = get_xy(b, Xvec)

        dx = xa - xb
        dy = ya - yb
        d  = np.hypot(dx, dy)
        w[k] = d_obs - d

        if d == 0:
            d = 1e-12
        dddx = dx / d
        dddy = dy / d

        # preenche apenas as colunas das estações desconhecidas presentes
        if a in param_index:
            ia, ja = param_index[a]
            A[k, ia] +=  dddx
            A[k, ja] +=  dddy
        if b in param_index:
            ib, jb = param_index[b]
            A[k, ib] += -dddx
            A[k, jb] += -dddy

    return w, A

# =========================
# 4) ITERAÇÃO (GAUSS-NEWTON)
# =========================
tol = 1e-10
max_it = 50

for it in range(1, max_it+1):
    w, A = build_w_A(X)
    N = A.T @ P @ A
    t = A.T @ P @ w

    try:
        dX = np.linalg.solve(N, t)
    except np.linalg.LinAlgError:
        dX = np.linalg.lstsq(N, t, rcond=None)[0]

    X = X + dX
    if np.linalg.norm(dX, 2) < tol:
        break

# =========================
# 5) PÓS-AJUSTE
# =========================
w, A = build_w_A(X)
v = -w                                         # v = A*0 - w (no final ΔX≈0)
m = len(obs)
nu = m - n                                     # graus de liberdade
sigma0_2 = (v.T @ P @ v) / nu

N = A.T @ P @ A
try:
    Cx = sigma0_2 * np.linalg.inv(N)
except np.linalg.LinAlgError:
    Cx = sigma0_2 * np.linalg.pinv(N)
sx = np.sqrt(np.diag(Cx))                      # desvios-padrão por coordenada

# =========================
# 6) SAÍDA
# =========================
print(f"Convergiu em {it} iterações")
for name in unk_names:
    i,j = param_index[name]
    print(f"{name}:  X = {X[i]:.4f} ± {sx[i]:.4f} m   |   Y = {X[j]:.4f} ± {sx[j]:.4f} m")

print(f"\nσ0² (a posteriori) = {sigma0_2:.6f}   |   ν = {nu}")
print("\nResíduos v (m):")
for k,(a,b,_,_) in enumerate(obs):
    print(f"{a}-{b}: {v[k]:+.4f}")





from scipy.stats import chi2

alpha = 0.05          # nível de significância (95%)
chi2_inf = chi2.ppf(alpha/2, nu)
chi2_sup = chi2.ppf(1 - alpha/2, nu)
chi2_calc = nu * sigma0_2

print("\nTeste do Qui-Quadrado (95%):")
print(f"χ²_calc = {chi2_calc:.3f}")
print(f"Intervalo aceitável: [{chi2_inf:.3f}, {chi2_sup:.3f}]")

if chi2_inf <= chi2_calc <= chi2_sup:
    print("✅ Ajuste compatível com as precisões a priori (aceito).")
else:
    print("⚠️ Ajuste rejeitado — há inconsistência com as precisões declaradas.")








# =========================
# 7) ELI PSES DE ERRO
# =========================
print("\nElipses de erro (1σ):")

for name in unk_names:
    i, j = param_index[name]
    # submatriz 2x2 de covariância do ponto
    Csub = Cx[np.ix_([i, j], [i, j])]

    # componentes
    sxx = Csub[0, 0]
    syy = Csub[1, 1]
    sxy = Csub[0, 1]

    # desvios padrão em x e y
    sx = np.sqrt(sxx)
    sy = np.sqrt(syy)

    # autovalores e autovetores (para eixos principais)
    eigvals, eigvecs = np.linalg.eig(Csub)
    a = np.sqrt(max(eigvals))  # semi-eixo maior
    b = np.sqrt(min(eigvals))  # semi-eixo menor

    # azimute do eixo maior (em graus)
    alpha = 0.5 * np.degrees(np.arctan2(2 * sxy, sxx - syy))

    print(f"{name}: σx = {sx:.4f} m, σy = {sy:.4f} m, "
          f"a = {a:.4f} m, b = {b:.4f} m, α = {alpha:.2f}°")

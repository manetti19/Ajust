# ============================================================
# AJUSTAMENTO GNSS (Rede 3D) + ITERATIVE DATA SNOOPING (IDS)
# - BEPA fixo (injunção fixa): X, Y, Z conhecidos (IBGE / RBMC)
# - Observações: vetores de linha de base (ΔX, ΔY, ΔZ)
# - Precisão dada (mm) => sigma por componente (m) => Σ_ll e pesos
# - IDS: w_i = v_i / sigma_{v_i}
# - sigma_{v_i} calculado pela LEI DE PROPAGAÇÃO:
#       Σ_vv = B Σ_ll B^T, com v = B l
# ============================================================

import numpy as np  # Biblioteca para vetores/matrizes e álgebra linear

# ------------------------------------------------------------
# 1) ENTRADAS FIXAS DO PROBLEMA
# ------------------------------------------------------------

# --- 1.1) Coordenadas fixas (injeção) da estação BEPA (ECEF) ---
# Fonte: Descritivo_BEPA.pdf (IBGE / RBMC)  (SIRGAS2000 época 2000.4)
X_BEPA = 4229786.5324   # (m) componente X fixa da BEPA
Y_BEPA = -4771063.6244  # (m) componente Y fixa da BEPA
Z_BEPA = -161510.2200   # (m) componente Z fixa da BEPA

BEPA_XYZ = np.array([X_BEPA, Y_BEPA, Z_BEPA], dtype=float)  # (m) vetor [X,Y,Z] de BEPA

# --- 1.2) Valor crítico do IDS (dado no enunciado) ---
CRITICO = 3.29  # critério: se max(|w|) > 3.29 remove e itera; caso contrário para

# --- 1.3) Pontos desconhecidos (incógnitas) ---
unknown_points = ["M01", "M02", "M03"]  # pontos cujas coordenadas serão estimadas

# Mapeamento: posição inicial de cada ponto no vetor de incógnitas x̂ (9 parâmetros)
# x̂ = [X_M01, Y_M01, Z_M01, X_M02, Y_M02, Z_M02, X_M03, Y_M03, Z_M03]^T
idx = {  # dicionário para localizar rapidamente índices no vetor x̂
    "M01": 0,
    "M02": 3,
    "M03": 6,
}

# --- 1.4) Linhas de base observadas (ΔX, ΔY, ΔZ) e precisão (mm) ---
# Interpretação: ΔX = X_to - X_fr  (idem para Y e Z)
baselines = [
    ("BEPA", "M01",  7849.9087,  3085.7272,  1505.4325, 13.80),
    ("M01",  "M02",  5118.6087,   576.9179,  3131.5131, 16.68),
    ("M02",  "M03", -6554.1662,  4284.0852,   223.2862, 13.94),
    ("BEPA", "M02", 12968.5476,  3662.5475,  4636.9275, 17.78),
    ("M01",  "M03", -1435.5490,  4860.9308,  3354.8164, 20.65),
    ("M03",  "BEPA", -6414.3606, -7946.6716, -4860.2321, 20.42),
]

# ------------------------------------------------------------
# 2) FUNÇÕES AUXILIARES (montagem do sistema, MMQ, IDS)
# ------------------------------------------------------------

def build_system(active_mask):
    """
    Monta A, l e Σ_ll (e P) para o MMQ com base nas observações ativas.
    - active_mask: vetor booleano do conjunto TOTAL (3 por baseline)
      True = observação entra no ajuste; False = removida.
    Retorna:
      A (m_active x 9), l (m_active,), Σ_ll (m_active x m_active), P (m_active x m_active),
      sigmas (m_active,), obs_meta (identificação de cada linha).
    """

    A_rows = []     # lista para acumular linhas da matriz A
    l_vals = []     # lista para acumular valores do vetor l
    sig_vals = []   # lista para acumular sigmas (m) por observação ativa
    obs_meta = []   # lista para rastrear qual baseline/componente gerou cada linha

    global_obs_index = 0  # índice no conjunto TOTAL (18 no seu caso: 6 baselines * 3 componentes)

    for b_id, (fr, to, dX, dY, dZ, sigma_mm) in enumerate(baselines, start=1):
        sigma_m = sigma_mm / 1000.0  # converte mm -> m (mesmo sigma para dX,dY,dZ da baseline)

        d_components = [dX, dY, dZ]              # valores observados (m)
        comp_names = ["dX", "dY", "dZ"]          # nomes das componentes

        for comp in range(3):
            # Se a observação foi removida, pula (mas avança o índice global)
            if not active_mask[global_obs_index]:
                global_obs_index += 1
                continue

            # Linha de A com 9 incógnitas (3 coords * 3 pontos)
            row = np.zeros(9, dtype=float)

            # Valor observado da componente atual
            l = float(d_components[comp])

            # ----------------------------
            # Modelo: (Coord_to - Coord_fr) = ΔCoord_obs + v
            # Se um dos pontos for BEPA (fixo), entra como constante no lado direito (l).
            # ----------------------------

            # Trata ponto "to"
            if to == "BEPA":
                # (Coord_BEPA - Coord_fr) = Δ
                # -> -Coord_fr = Δ - Coord_BEPA
                l = l - BEPA_XYZ[comp]  # move constante para o lado direito
            else:
                # Coord_to é incógnita: entra com +1
                row[idx[to] + comp] += 1.0

            # Trata ponto "fr"
            if fr == "BEPA":
                # (Coord_to - Coord_BEPA) = Δ
                # -> Coord_to = Δ + Coord_BEPA
                l = l + BEPA_XYZ[comp]  # move constante para o lado direito
            else:
                # Coord_fr é incógnita: entra com -1
                row[idx[fr] + comp] -= 1.0

            # Guarda linha e observação
            A_rows.append(row)
            l_vals.append(l)

            # Guarda sigma (m) dessa observação
            sig_vals.append(sigma_m)

            # Guarda metadado para rastrear depois
            obs_meta.append((b_id, comp_names[comp], fr, to))

            # Avança índice no conjunto total
            global_obs_index += 1

    # Converte listas em arrays/matrizes
    A = np.vstack(A_rows)                     # (m_active x 9)
    l = np.array(l_vals, dtype=float)         # (m_active,)
    sigmas = np.array(sig_vals, dtype=float)  # (m_active,)

    # Matriz variância-covariância das observações (descorr.): Σ_ll = diag(sigmas^2)
    Sigma_ll = np.diag(sigmas * sigmas)       # (m_active x m_active)

    # Matriz de pesos: P = Σ_ll^-1
    P = np.diag(1.0 / (sigmas * sigmas))      # diagonal (mais rápido do que inv geral)

    return A, l, Sigma_ll, P, sigmas, obs_meta


def least_squares_with_propagation(A, l, Sigma_ll, P):
    """
    Resolve o MMQ ponderado e calcula a variância dos resíduos por propagação.
    Retorna:
      x_hat, v, Sigma_xx, Sigma_vv_prop, sigma_v, sigma0_sq
    """

    # Matriz normal: N = A^T P A
    N = A.T @ P @ A  # (9x9)

    # Termo independente: n = A^T P l
    n_vec = A.T @ P @ l  # (9,)

    # Solução estimada: x̂ = N^-1 n
    x_hat = np.linalg.solve(N, n_vec)  # (9,)

    # Resíduos: v = A x̂ - l
    v = (A @ x_hat) - l  # (m,)

    # Graus de liberdade: nu = m - u
    m = A.shape[0]  # número de observações ativas
    u = A.shape[1]  # número de incógnitas (9)
    nu = m - u      # graus de liberdade

    # Variância a posteriori (se você quiser escalar):
    # sigma0^2 = (v^T P v)/nu
    sigma0_sq = float((v.T @ P @ v) / nu)

    # Covariância das incógnitas (modelo linear): Σ_xx = (A^T Σ_ll^-1 A)^-1 = (A^T P A)^-1
    Sigma_xx = np.linalg.inv(N)  # (9x9)

    # Matriz G tal que x̂ = G l
    # G = (A^T P A)^-1 A^T P
    G = Sigma_xx @ (A.T @ P)  # (u x m)

    # Matriz B tal que v = B l
    # v = A x̂ - l = (A G - I) l
    B = (A @ G) - np.eye(m)  # (m x m)

    # Lei de propagação: Σ_vv = B Σ_ll B^T
    Sigma_vv_prop = B @ Sigma_ll @ B.T  # (m x m)

    # Desvio-padrão de cada resíduo: sigma_v_i = sqrt( diag(Σ_vv) )
    sigma_v = np.sqrt(np.diag(Sigma_vv_prop))  # (m,)

    return x_hat, v, Sigma_xx, Sigma_vv_prop, sigma_v, sigma0_sq


def iterative_data_snooping(max_loops=20, remove_entire_baseline=False):
    """
    IDS:
    - Ajusta por MMQ
    - Calcula sigma_v pelos resíduos via propagação: Σ_vv = B Σ_ll B^T
    - Calcula w_i = v_i / sigma_v_i
    - Remove a observação (ou baseline inteira) com maior |w| se |w| > CRITICO
    """

    m_total = 3 * len(baselines)  # total bruto: 3 por baseline
    active_mask = np.ones(m_total, dtype=bool)  # começa usando tudo
    removed = []  # lista de remoções

    for loop in range(1, max_loops + 1):
        # Monta o sistema atual
        A, l, Sigma_ll, P, sigmas, obs_meta = build_system(active_mask)

        # Checagem: precisa m > u
        if A.shape[0] <= A.shape[1]:
            raise RuntimeError("Poucas observações restantes: não dá para ajustar (m <= número de incógnitas).")

        # Ajusta e calcula sigma_v via propagação
        x_hat, v, Sigma_xx, Sigma_vv_prop, sigma_v, sigma0_sq = least_squares_with_propagation(A, l, Sigma_ll, P)

        # Estatística IDS
        w = v / sigma_v  # (m_active,)

        # Acha maior |w|
        k = int(np.argmax(np.abs(w)))
        w_max = float(w[k])
        w_abs = abs(w_max)

        # Identifica a observação "pior"
        b_id, comp_name, fr, to = obs_meta[k]

        print(f"[Loop {loop:02d}] pior obs: baseline {b_id} ({fr}->{to}) {comp_name} | w = {w_max:+.3f} | |w| = {w_abs:.3f}")

        # Se não excede o crítico, para
        if w_abs <= CRITICO:
            info_final = {
                "A": A,
                "l": l,
                "P": P,
                "Sigma_ll": Sigma_ll,
                "Sigma_xx": Sigma_xx,
                "Sigma_vv": Sigma_vv_prop,
                "sigmas_obs": sigmas,
                "sigma_v": sigma_v,
                "v": v,
                "w": w,
                "sigma0": float(np.sqrt(sigma0_sq)),
                "obs_meta": obs_meta,
            }
            return x_hat, removed, info_final

        # Caso exceda: registra e remove
        removed.append((b_id, comp_name, fr, to, w_max))

        # Descobre índice global (no conjunto TOTAL) da observação identificada
        global_index_to_remove = None
        g = 0
        for bb_id, (fr2, to2, _, _, _, _) in enumerate(baselines, start=1):
            for compn2 in ["dX", "dY", "dZ"]:
                if (bb_id == b_id) and (compn2 == comp_name):
                    global_index_to_remove = g
                g += 1

        if global_index_to_remove is None:
            raise RuntimeError("Não consegui mapear a observação ativa para o índice global.")

        # Remove componente apenas
        if not remove_entire_baseline:
            active_mask[global_index_to_remove] = False

        # Remove baseline inteira (3 componentes)
        if remove_entire_baseline:
            base_start = (b_id - 1) * 3
            active_mask[base_start + 0] = False
            active_mask[base_start + 1] = False
            active_mask[base_start + 2] = False

    raise RuntimeError("IDS não convergiu dentro do número máximo de iterações.")


def unpack_solution(x_hat):
    """
    Converte x̂ (9,) para coordenadas 3D dos pontos M01, M02, M03.
    """
    sol = {}
    for p in unknown_points:
        i0 = idx[p]
        sol[p] = np.array([x_hat[i0 + 0], x_hat[i0 + 1], x_hat[i0 + 2]], dtype=float)
    return sol


# ------------------------------------------------------------
# 3) EXECUÇÃO
# ------------------------------------------------------------

REMOVE_BASELINE_TODA = False  # True = remove baseline inteira; False = remove só componente

x_final, outliers, info = iterative_data_snooping(remove_entire_baseline=REMOVE_BASELINE_TODA)

coords = unpack_solution(x_final)

# ------------------------------------------------------------
# 4) RELATÓRIO FINAL
# ------------------------------------------------------------

print("\n==================== OUTLIERS REMOVIDOS (IDS) ====================")
if len(outliers) == 0:
    print("Nenhuma observação foi classificada como outlier (|w| <= 3.29).")
else:
    for j, (b_id, comp_name, fr, to, w_val) in enumerate(outliers, start=1):
        print(f"{j:02d}) baseline {b_id} ({fr}->{to}) {comp_name} removida | w = {w_val:+.3f}")

print("\n==================== COORDENADAS FIXAS (BEPA) ====================")
print(f"BEPA: X = {X_BEPA:.4f} m | Y = {Y_BEPA:.4f} m | Z = {Z_BEPA:.4f} m")

print("\n==================== COORDENADAS AJUSTADAS (ECEF) ====================")
for p in unknown_points:
    Xp, Yp, Zp = coords[p]
    print(f"{p}: X = {Xp:.4f} m | Y = {Yp:.4f} m | Z = {Zp:.4f} m")

print("\n==================== QUALIDADE DO AJUSTAMENTO ====================")
print(f"sigma0 (final) = {info['sigma0']:.6f} (unidade compatível com o modelo)")

print("\n==================== RESÍDUOS, sigma_v e w (OBS ATIVAS) ====================")
v = info["v"]
sigma_v = info["sigma_v"]
w = info["w"]
obs_meta = info["obs_meta"]

for i in range(len(v)):
    b_id, comp_name, fr, to = obs_meta[i]
    print(f"baseline {b_id} ({fr}->{to}) {comp_name}: v = {v[i]:+.6f} m | sigma_v = {sigma_v[i]:.6f} m | w = {w[i]:+.3f}")

# ------------------------------------------------------------
# 5) PRINT DA MATRIZ A FINAL (COMO VOCÊ PEDIU)
# ------------------------------------------------------------

print("\n==================== MATRIZ A FINAL ====================")
A_final = info["A"]

np.set_printoptions(precision=1, suppress=True)  # aparência: 1 casa, sem notação científica
print("Dimensão de A:", A_final.shape)
print(A_final)

print("\n==================== MATRIZ A FINAL (COM IDENTIFICAÇÃO) ====================")
np.set_printoptions(precision=1, suppress=True)

for i in range(A_final.shape[0]):
    b_id, comp_name, fr, to = obs_meta[i]
    print(f"\nObs {i+1:02d} | baseline {b_id} ({fr}->{to}) {comp_name}")
    print(A_final[i])
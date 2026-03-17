# ============================================================
# AJUSTAMENTO GNSS (Rede 3D) + ITERATIVE DATA SNOOPING (IDS)
# - BEPA fixo (injunção fixa): X, Y, Z conhecidos (IBGE / RBMC)
# - Observações: vetores de linha de base (ΔX, ΔY, ΔZ)
# - Precisão dada (mm) => sigma por componente (m) => pesos
# - IDS: w_i = v_i / sigma_{v_i}   (observações descorrelacionadas)
# - Valor crítico fornecido: 3.29
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

# Agrupa BEPA em um vetor 3D (facilita contas)
BEPA_XYZ = np.array([X_BEPA, Y_BEPA, Z_BEPA], dtype=float)  # [X, Y, Z] da BEPA

# --- 1.2) Valor crítico do IDS (dado no enunciado) ---
CRITICO = 3.29  # valor crítico para comparar com max(|w_i|)

# --- 1.3) Nomes dos pontos desconhecidos (nós da rede) ---
# Aqui: M01, M02, M03 são incógnitas 3D (X,Y,Z)
unknown_points = ["M01", "M02", "M03"]  # lista com os nomes dos pontos desconhecidos

# Mapeia cada ponto desconhecido para a posição no vetor de incógnitas
# Vetor x = [X_M01, Y_M01, Z_M01, X_M02, Y_M02, Z_M02, X_M03, Y_M03, Z_M03]^T
idx = {  # dicionário para localizar rapidamente índices no vetor x
    "M01": 0,  # começa em 0
    "M02": 3,  # começa em 3
    "M03": 6,  # começa em 6
}

# --- 1.4) Tabela de linhas de base (dados do enunciado) ---
# Cada item tem: (de, para, dX, dY, dZ, sigma_mm)
# Interpretação: ΔX = X_para - X_de (mesma lógica para Y e Z)
baselines = [  # lista com todas as observações de linha de base
    ("BEPA", "M01",  7849.9087,  3085.7272,  1505.4325, 13.80),
    ("M01",  "M02",  5118.6087,   576.9179,  3131.5131, 16.68),
    ("M02",  "M03", -6554.1662,  4284.0852,   223.2862, 13.94),
    ("BEPA", "M02", 12968.5476,  3662.5475,  4636.9275, 17.78),
    ("M01",  "M03", -1435.5490,  4860.9308,  3354.8164, 20.65),
    ("M03",  "BEPA", -6414.3606, -7946.6716, -4860.2321, 20.42),
]

# ------------------------------------------------------------
# 2) FUNÇÕES AUXILIARES (MMQ + IDS)
# ------------------------------------------------------------

def build_system(active_mask):
    """
    Monta A, l e P para o MMQ com base em quais observações estão ativas.
    - active_mask: vetor booleano (m,), onde m = 3 * número_de_baselines
      True = observação entra no ajuste; False = removida (outlier).
    Retorna:
      A (m_active x 9), l (m_active,), P (m_active x m_active),
      obs_meta (lista com metadados da linha de A) para rastrear w_i e outliers.
    """

    # Vamos acumular linhas da matriz A e valores do vetor l em listas
    A_rows = []          # lista de linhas da matriz A
    l_vals = []          # lista de valores de observação (lado direito)
    w_sigs = []          # lista dos sigmas (m) de cada observação (para pesos)
    obs_meta = []        # lista com metadados: (baseline_id, componente, de, para)

    # Contador global de observações: cada baseline gera 3 observações (X, Y, Z)
    global_obs_index = 0  # índice que percorre o active_mask

    for b_id, (fr, to, dX, dY, dZ, sigma_mm) in enumerate(baselines, start=1):
        # Converte sigma de mm para m (um sigma igual para ΔX, ΔY, ΔZ daquela baseline)
        sigma_m = sigma_mm / 1000.0  # (m) desvio-padrão por componente

        # Empacota componentes para iterar: 0->X, 1->Y, 2->Z
        d_components = [dX, dY, dZ]  # valores observados para cada componente
        comp_names = ["dX", "dY", "dZ"]  # nomes para rastrear

        for comp in range(3):
            # Se essa observação foi marcada como removida, pula
            if not active_mask[global_obs_index]:
                global_obs_index += 1  # avança o índice global
                continue  # não inclui essa observação no sistema

            # Cria uma linha de A com 9 incógnitas (3 coords por ponto desconhecido)
            row = np.zeros(9, dtype=float)  # inicializa tudo com zero

            # Observação (lado direito) começa com o valor medido da componente (ΔX, ΔY ou ΔZ)
            l = float(d_components[comp])  # valor observado para a componente atual

            # --------------------------------------------------------
            # Modelo: (Coord_to - Coord_fr) = ΔCoord_observada + v
            #
            # Se um ponto é FIXO (BEPA), sua coordenada entra como constante no l.
            # Se um ponto é DESCONHECIDO (M01/M02/M03), entra na linha de A.
            # --------------------------------------------------------

            # ----- Trata o ponto "to" -----
            if to == "BEPA":
                # Coord_to é fixa: adiciona ( + Coord_BEPA ) no lado esquerdo.
                # Para manter A*x = l, passamos essa constante para o lado direito:
                # (Coord_to_fixa - Coord_fr_incognita) = Δ => -Coord_fr_incognita = Δ - Coord_to_fixa
                l = l - BEPA_XYZ[comp]  # move Coord_BEPA para o lado direito com sinal negativo
            else:
                # Coord_to é incógnita: entra com coeficiente +1 na coluna correspondente
                row[idx[to] + comp] += 1.0  # soma +1 na variável correta (X/Y/Z do ponto "to")

            # ----- Trata o ponto "fr" -----
            if fr == "BEPA":
                # Coord_fr é fixa: a expressão é (Coord_to_incognita - Coord_BEPA) = Δ
                # Passa (-Coord_BEPA) para o lado direito: Coord_to_incognita = Δ + Coord_BEPA
                l = l + BEPA_XYZ[comp]  # move Coord_BEPA para o lado direito com sinal positivo
            else:
                # Coord_fr é incógnita: entra com coeficiente -1
                row[idx[fr] + comp] -= 1.0  # subtrai 1 na variável correta (X/Y/Z do ponto "fr")

            # Guarda a linha e o valor l no sistema
            A_rows.append(row)  # adiciona linha na lista
            l_vals.append(l)    # adiciona observação ajustada (com constantes já movidas)

            # Guarda sigma dessa observação para compor a matriz de pesos
            w_sigs.append(sigma_m)  # guarda sigma em metros

            # Guarda metadados para saber "quem é quem" depois
            obs_meta.append((b_id, comp_names[comp], fr, to))  # identifica baseline e componente

            # Avança índice global de observações
            global_obs_index += 1  # próxima observação

    # Converte listas em arrays
    A = np.vstack(A_rows)  # empilha as linhas e vira matriz (m_active x 9)
    l = np.array(l_vals, dtype=float)  # vetor (m_active,)
    sigmas = np.array(w_sigs, dtype=float)  # vetor (m_active,)

    # Matriz de pesos P (diagonal) com p_i = 1/sigma_i^2
    P = np.diag(1.0 / (sigmas * sigmas))  # matriz diagonal (m_active x m_active)

    return A, l, P, sigmas, obs_meta  # retorna tudo necessário para o MMQ


def least_squares(A, l, P):
    """
    Resolve o MMQ ponderado:
      x̂ = (A^T P A)^(-1) A^T P l
    Retorna x̂, resíduos v, sigma0^2, Qxx e Qvv (para sigma_v).
    """

    # Matriz normal N = A^T P A
    N = A.T @ P @ A  # (9x9)

    # Vetor dos termos independentes n = A^T P l
    n_vec = A.T @ P @ l  # (9,)

    # Resolve o sistema normal: N x̂ = n
    x_hat = np.linalg.solve(N, n_vec)  # solução estimada (9,)

    # Resíduos: v = A x̂ - l
    v = (A @ x_hat) - l  # vetor (m,)

    # Graus de liberdade: ν = m - u
    m = A.shape[0]       # número de observações usadas
    u = A.shape[1]       # número de incógnitas (9)
    nu = m - u           # graus de liberdade

    # Variância a posteriori: sigma0^2 = (v^T P v) / ν
    sigma0_sq = float((v.T @ P @ v) / nu)  # escalar

    # Matriz de cofatores das incógnitas: Qxx = (A^T P A)^(-1)
    Qxx = np.linalg.inv(N)  # (9x9)

    # Matriz de cofatores das observações: Qll = P^-1
    Qll = np.linalg.inv(P)  # (m x m) diagonal

    # Matriz de cofatores dos resíduos: Qvv = Qll - A Qxx A^T
    Qvv = Qll - (A @ Qxx @ A.T)  # (m x m)

    return x_hat, v, sigma0_sq, Qxx, Qvv  # devolve resultados úteis


def iterative_data_snooping(max_loops=20, remove_entire_baseline=False):
    """
    Faz IDS:
    - Ajusta
    - Calcula w_i = v_i / sigma_v_i
    - Se max|w| > CRITICO remove a observação e repete
    Parâmetro remove_entire_baseline:
      - False: remove apenas a componente (dX/dY/dZ) que estourou
      - True : remove as 3 componentes daquela baseline (mais conservador)
    Retorna:
      x_final, lista_outliers, info_final
    """

    # Total de observações brutas: 3 por baseline
    m_total = 3 * len(baselines)  # aqui: 3*6 = 18

    # Máscara indicando quais observações estão ativas (True = usa no ajuste)
    active_mask = np.ones(m_total, dtype=bool)  # começa com tudo True

    # Lista para registrar outliers removidos em ordem
    removed = []  # cada item: (baseline_id, comp, fr, to, w_value)

    for loop in range(1, max_loops + 1):
        # Monta sistema (A, l, P) apenas com observações ativas
        A, l, P, sigmas, obs_meta = build_system(active_mask)  # constroi o sistema atual

        # Checagem simples: não dá para ajustar se m <= u
        if A.shape[0] <= A.shape[1]:
            raise RuntimeError("Poucas observações restantes: não dá para ajustar (m <= número de incógnitas).")

        # Resolve MMQ
        x_hat, v, sigma0_sq, Qxx, Qvv = least_squares(A, l, P)  # estima incógnitas e resíduos

        # Desvio-padrão dos resíduos: sigma_v_i = sqrt( sigma0^2 * q_vv_ii )
        qvv_diag = np.diag(Qvv)  # pega apenas a diagonal de Qvv (um por observação)
        sigma_v = np.sqrt(sigma0_sq * qvv_diag)  # vetor (m_ativo,)

        # Estatística IDS por observação: w_i = v_i / sigma_v_i
        w = v / sigma_v  # vetor (m_ativo,)

        # Encontra o índice do maior |w|
        k = int(np.argmax(np.abs(w)))  # índice na lista ATIVA
        w_max = float(w[k])            # valor (com sinal)
        w_abs = abs(w_max)             # valor absoluto

        # Imprime uma linha de acompanhamento (opcional)
        b_id, comp_name, fr, to = obs_meta[k]  # identifica a observação "pior"
        print(f"[Loop {loop:02d}] pior obs: baseline {b_id} ({fr}->{to}) {comp_name} | w = {w_max:+.3f} | |w| = {w_abs:.3f}")

        # Se NÃO ultrapassa o crítico, para (rede limpa)
        if w_abs <= CRITICO:
            # Monta informações finais para saída
            info_final = {
                "sigma0": float(np.sqrt(sigma0_sq)),  # sigma0 final
                "v": v,                               # resíduos finais (apenas das observações ativas)
                "w": w,                               # w finais (apenas das observações ativas)
                "obs_meta": obs_meta,                 # metadados das observações ativas
                "A": A,                               # matriz A final
                "l": l,                               # vetor l final
                "P": P,                               # pesos finais
            }
            return x_hat, removed, info_final  # retorna solução final e outliers removidos

        # Se ultrapassa o crítico, marca como outlier e remove
        removed.append((b_id, comp_name, fr, to, w_max))  # registra a remoção

        # Agora precisamos “desligar” essa observação no active_mask (que é do conjunto TOTAL)
        # Para isso, descobrimos qual é o índice global daquela observação no conjunto total.
        # Estratégia: reconstruir o mapeamento global (baseline->3 componentes) e achar o match.
        global_index_to_remove = None  # inicializa como None

        # Índice global vai de 0 até 17 (18 obs)
        g = 0  # contador global
        for bb_id, (fr2, to2, _, _, _, _) in enumerate(baselines, start=1):
            for comp2, compn2 in enumerate(["dX", "dY", "dZ"]):
                # Se essa observação já está removida, apenas avança
                # (ainda assim o índice global precisa avançar)
                # A comparação aqui precisa achar o mesmo baseline e componente
                if (bb_id == b_id) and (compn2 == comp_name):
                    global_index_to_remove = g  # achou o índice global da observação
                g += 1  # avança global

        # Segurança: se não achou, erro
        if global_index_to_remove is None:
            raise RuntimeError("Não consegui mapear a observação ativa para o índice global.")

        # Remove apenas a componente (modo padrão)
        if not remove_entire_baseline:
            active_mask[global_index_to_remove] = False  # remove somente aquela componente

        # Ou remove as 3 componentes da baseline (modo conservador)
        if remove_entire_baseline:
            # Calcula os 3 índices globais dessa baseline:
            # baseline 1 ocupa [0,1,2], baseline 2 ocupa [3,4,5], etc.
            base_start = (b_id - 1) * 3  # início do bloco da baseline
            active_mask[base_start + 0] = False  # remove dX
            active_mask[base_start + 1] = False  # remove dY
            active_mask[base_start + 2] = False  # remove dZ

    # Se estourar número máximo de loops, retorna o que tiver (ou lança erro)
    raise RuntimeError("IDS não convergiu dentro do número máximo de iterações.")


def unpack_solution(x_hat):
    """
    Converte o vetor x̂ (9,) para coordenadas 3D dos pontos M01, M02, M03.
    Retorna um dicionário com XYZ.
    """
    sol = {}  # dicionário de saída
    for p in unknown_points:  # percorre M01, M02, M03
        i0 = idx[p]  # índice inicial do ponto no vetor
        sol[p] = np.array([x_hat[i0 + 0], x_hat[i0 + 1], x_hat[i0 + 2]], dtype=float)  # [X,Y,Z]
    return sol  # retorna as coordenadas


# ------------------------------------------------------------
# 3) EXECUÇÃO DO IDS + AJUSTAMENTO FINAL
# ------------------------------------------------------------

# Define se você quer remover só a componente (False) ou a baseline toda (True)
REMOVE_BASELINE_TODA = False  # mude para True se o professor quiser remover a baseline inteira

# Roda o IDS e pega a solução final
x_final, outliers, info = iterative_data_snooping(remove_entire_baseline=REMOVE_BASELINE_TODA)  # executa IDS

# Desempacota coordenadas estimadas
coords = unpack_solution(x_final)  # transforma x_final em XYZ para cada ponto

# ------------------------------------------------------------
# 4) RELATÓRIO FINAL (prints)
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
    Xp, Yp, Zp = coords[p]  # pega XYZ do ponto
    print(f"{p}: X = {Xp:.4f} m | Y = {Yp:.4f} m | Z = {Zp:.4f} m")

print("\n==================== QUALIDADE DO AJUSTAMENTO ====================")
print(f"sigma0 (final) = {info['sigma0']:.4f} m")

print("\n==================== RESÍDUOS E w (OBS ATIVAS) ====================")
v = info["v"]         # resíduos finais
w = info["w"]         # w finais
obs_meta = info["obs_meta"]  # metadados das observações ativas


for i in range(len(v)):
    b_id, comp_name, fr, to = obs_meta[i]  # identifica observação
    print(f"baseline {b_id} ({fr}->{to}) {comp_name}: v = {v[i]:+.4f} m | w = {w[i]:+.3f}")


print("\n==================== MATRIZ A FINAL ====================")

A_final = info["A"]  # matriz A após IDS (já sem observações removidas)

np.set_printoptions(precision=6, suppress=True)  # melhora visualização

print("Dimensão de A:", A_final.shape)
print(A_final)
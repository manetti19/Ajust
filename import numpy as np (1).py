import numpy as np


MEDIA_VERDADEIRA = 10
N_MEDIDAS = 1000
DP_1 = 0.5
DP_2 = 1.0
SEED = 42


def resumo_amostra(nome, observacoes, media_verdadeira):
    media_amostral = np.mean(observacoes)
    variancia_amostral = np.var(observacoes, ddof=1)
    variancia_da_media = variancia_amostral / len(observacoes)
    erro = media_amostral - media_verdadeira

    print(f"{nome}")
    print(f"  media amostral           = {media_amostral:.6f}")
    print(f"  valor verdadeiro         = {media_verdadeira:.6f}")
    print(f"  erro da estimativa       = {erro:.6f}")
    print(f"  variancia amostral       = {variancia_amostral:.6f}")
    print(f"  variancia da media       = {variancia_da_media:.8f}")
    print()

    return {
        "media": media_amostral,
        "variancia_amostral": variancia_amostral,
        "variancia_da_media": variancia_da_media,
        "erro": erro,
    }


def media_ponderada(obs1, obs2, dp1, dp2):
    # Pesos de precisao: observacoes com menor variancia recebem maior peso.
    peso1 = 1 / (dp1 ** 2)
    peso2 = 1 / (dp2 ** 2)

    soma_ponderada = peso1 * np.sum(obs1) + peso2 * np.sum(obs2)
    soma_pesos = peso1 * len(obs1) + peso2 * len(obs2)
    estimativa = soma_ponderada / soma_pesos

    variancia_teorica = 1 / soma_pesos
    todas = np.concatenate([obs1, obs2])
    pesos = np.concatenate([
        np.full(len(obs1), peso1),
        np.full(len(obs2), peso2),
    ])

    media_pesos = np.average(todas, weights=pesos)
    variancia_ponderada_amostral = np.average((todas - media_pesos) ** 2, weights=pesos)

    print("3) Combinacao ponderada das observacoes")
    print(f"  peso usado no grupo 1    = {peso1:.2f}")
    print(f"  peso usado no grupo 2    = {peso2:.2f}")
    print(f"  media ponderada          = {estimativa:.6f}")
    print(f"  valor verdadeiro         = {MEDIA_VERDADEIRA:.6f}")
    print(f"  erro da estimativa       = {estimativa - MEDIA_VERDADEIRA:.6f}")
    print(f"  variancia teorica media  = {variancia_teorica:.8f}")
    print(f"  variancia ponderada obs  = {variancia_ponderada_amostral:.6f}")
    print()

    return {
        "media_ponderada": estimativa,
        "variancia_teorica_media": variancia_teorica,
        "variancia_ponderada_amostral": variancia_ponderada_amostral,
    }


def analisar_resultados(r1, r2, r3):
    print("Analise final")
    print(
        "  O grupo 1 tende a produzir uma media mais precisa, porque seu desvio-padrao"
        f" e menor ({DP_1}) que o do grupo 2 ({DP_2})."
    )
    print(
        "  A combinacao ponderada atribui maior peso ao grupo 1 e, por isso,"
        " a variancia da media combinada tende a ser menor do que usar apenas o grupo 2."
    )
    print(
        "  Comparacao das variancias da media:"
        f" grupo 1 = {r1['variancia_da_media']:.8f},"
        f" grupo 2 = {r2['variancia_da_media']:.8f},"
        f" ponderada = {r3['variancia_teorica_media']:.8f}"
    )


def main():
    np.random.seed(SEED)

    obs1 = np.random.normal(MEDIA_VERDADEIRA, DP_1, N_MEDIDAS)
    obs2 = np.random.normal(MEDIA_VERDADEIRA, DP_2, N_MEDIDAS)

    print("1) Simulacao com media=10 e DP=0.5")
    resultado_1 = resumo_amostra("  Resultados do grupo 1", obs1, MEDIA_VERDADEIRA)

    print("2) Simulacao com media=10 e DP=1.0")
    resultado_2 = resumo_amostra("  Resultados do grupo 2", obs2, MEDIA_VERDADEIRA)

    resultado_3 = media_ponderada(obs1, obs2, DP_1, DP_2)
    analisar_resultados(resultado_1, resultado_2, resultado_3)


if __name__ == "__main__":
    main()

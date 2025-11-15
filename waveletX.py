# ===============================================
# Leitura e filtragem Wavelet de dados de IMU
# ===============================================

import pandas as pd
import pywt
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------
# 1. Leitura do arquivo TXT da IMU
# -----------------------------------------------

arquivo = r"C:\Users\dinos\Downloads\Dados_Crossbow300_5h.txt"

colunas = [
    "Time (s)",
    "Roll rate (deg/s)",
    "Pitch rate (deg/s)",
    "Yaw rate (deg/s)",
    "X Accel (g)",
    "Y Accel (g)",
    "Z Accel (g)",
    "Temperature (°C)",
    "Timer counts"
]

df = pd.read_csv(
    arquivo,
    sep="\t",
    decimal=",",
    encoding="latin1"
)

# Ordena pelo tempo real
# df = df.sort_values("Time (s)")

# -----------------------------------------------
# 2. Selecionar e reduzir sinal
# -----------------------------------------------

# Aceleração no eixo X
sinal = df["X Accel (g)"].values
# tempo = df["Time (s)"].values

# Tempo mínimo e máximo reais (do arquivo original)
t_min = 2.964309
t_max = 19527.994141

N = 1952504

# Cria eixo de tempo artificial uniformemente espaçado
tempo = np.linspace(t_min, t_max, N)


# -----------------------------------------------
# 3. Wavelet Discreta (DWT) — Denoising
# -----------------------------------------------

wavelet = 'db4'
nivel = 4

coeficientes = pywt.wavedec(sinal, wavelet, level=nivel)
limiar = np.std(coeficientes[-1]) * np.sqrt(2 * np.log(len(sinal)))
coef_filtrados = [coeficientes[0]] + [
    pywt.threshold(c, limiar, mode='soft') for c in coeficientes[1:]
]
sinal_filtrado = pywt.waverec(coef_filtrados, wavelet)

# Ajusta tamanho
min_len = min(len(sinal), len(sinal_filtrado))
sinal = sinal[:min_len]
sinal_filtrado = sinal_filtrado[:min_len]
# tempo = tempo[:min_len]

# -----------------------------------------------
# 4. Plotagem
# -----------------------------------------------

plt.figure(figsize=(18, 6))
plt.plot(tempo, sinal, label="Sinal Original", alpha=0.4, linewidth=1)
plt.plot(tempo, sinal_filtrado, label="Sinal Filtrado (Wavelet)", linewidth=1)
plt.title("Filtragem de Ruído - X Accel (g)")
plt.xlabel("Tempo (s)")
plt.ylabel("Aceleração")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Tempo mínimo:", tempo.min())
print("Tempo máximo:", tempo.max())

print("len tempo =", len(tempo))
print("len sinal =", len(sinal))


df_out = pd.DataFrame({
    "tempo_s": tempo,
    "x_original": sinal,
    "x_filtrado": sinal_filtrado
})

df_out.to_csv("saida_wavelet_X.csv", index=False)
df_out.to_csv("saida_wavelet_X.txt", sep="\t", index=False)

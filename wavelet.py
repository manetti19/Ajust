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
    skiprows=8,
    encoding="latin1",
    names=colunas
)

# Ordena pelo tempo real
df = df.sort_values("Time (s)").reset_index(drop=True)

# -----------------------------------------------
# 2. Selecionar e reduzir sinal
# -----------------------------------------------

# Aceleração no eixo X
sinal = df["X Accel (g)"].values
tempo = df["Time (s)"].values



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
tempo = tempo[:min_len]

# -----------------------------------------------
# 4. Plotagem
# -----------------------------------------------

plt.figure(figsize=(12, 6))
plt.plot(tempo, sinal, label="Sinal Original", alpha=0.4)
plt.plot(tempo, sinal_filtrado, label="Sinal Filtrado (Wavelet)", linewidth=2)
plt.title("Filtragem de Ruído - X Accel (g)")
plt.xlabel("Tempo (s)")
plt.ylabel("Aceleração")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Tempo mínimo:", tempo.min())
print("Tempo máximo:", tempo.max())
# -*- coding: utf-8 -*-

# Número de lances necessários
n_lances = 10       # 500 m / 50 m = 10

# Desvio-padrão máximo tolerado no total D
sigma_D = 0.010     # 10 mm = 0,010 m

# Fórmula da propagação: sigma_D^2 = n * sigma^2
# Logo: sigma = sigma_D / sqrt(n)
import math
sigma_lance = sigma_D / math.sqrt(n_lances)

# Impressão dos resultados
print(f"Número de lances: {n_lances}")
print(f"Desvio-padrão máximo em D: {sigma_D:.3f} m")
print(f"Precisão mínima necessária em cada lance: {sigma_lance:.6f} m")

# Também em milímetros
print(f"≈ {sigma_lance*1000:.2f} mm por lance")

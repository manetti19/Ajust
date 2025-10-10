import numpy as np

# Dados fornecidos
comprimentos = np.array([2, 3, 2, 4, 5, 4])  # Comprimento das linhas (em km)
desnivel = np.array([6.16, 12.57, 6.41, 1.09, 11.58, 5.07])  # Desnível das linhas (em metros)

# Cálculo dos pesos (inversamente proporcionais ao comprimento)
pesos = 1 / comprimentos

# Matriz A (coeficientes das equações)
# O sistema de equações será baseado nas relações de desnível entre as estações
# Para o exemplo, precisamos montar a matriz de coeficientes A (onde temos 3 estações B, C, D)
# De acordo com as equações de nivelamento que representam as diferenças entre as altitudes
A = np.array([[-1, 1, 0],
              [0, -1, 1],
              [-1, 0, 1],
              [0, 1, -1],
              [1, 0, -1],
              [0, -1, 0]])

# Vetor de observações (desnível)
b = desnivel

# Matriz de pesos (ponderação por linha de observação)
W = np.diag(pesos)

# Resolução do sistema utilizando o método dos Mínimos Quadrados
# Ajuste das alturas das estações B, C e D
# x = (A.T @ W @ A)^(-1) @ (A.T @ W @ b)
x = np.linalg.inv(A.T @ W @ A) @ (A.T @ W @ b)

# Resultados (altitudes ajustadas)
print("Altitudes ajustadas (em metros) para as estações B, C e D:")
print(x)

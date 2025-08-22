p = print
import math 
import numpy as np
from numpy.linalg import inv, det, eig, svd, norm, solve

L = np.array([[57.3, 57.5, 57.4, 57.1, 57.4],
              [72.4, 72.3, 72.1, 72.3, 72.5],
              [36.8, 36.8, 37.1, 36.9, 37.0]])

def media(valores):
    return sum(valores) / len(valores)

def variancia(valores):
    n = len(valores)
    m = media(valores)
    return sum((x - m)**2 for x in valores) / (n - 1)

def covariancia(x, y):
    if len(x) != len(y):
        raise ValueError("As listas precisam ter o mesmo tamanho")
    n = len(x)
    mx, my = media(x), media(y)
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n - 1)


# Exemplo de uso com os dados da questão:
l1 = [57.3, 57.5, 57.4, 57.1, 57.4]
l2 = [72.4, 72.3, 72.1, 72.3, 72.5]
l3 = [36.8, 36.8, 37.1, 36.9, 37.0]

print("Var(l1):", variancia(l1))
print("Var(l2):", variancia(l2))
print("Var(l3):", variancia(l3))

print("Cov(l1,l2):", covariancia(l1, l2))
print("Cov(l1,l3):", covariancia(l1, l3))
print("Cov(l2,l3):", covariancia(l2, l3))


# Montar a matriz variância-covariância
Sigma = np.array([
    [variancia(l1), covariancia(l1, l2), covariancia(l1, l3)],
    [covariancia(l2, l1), variancia(l2), covariancia(l2, l3)],
    [covariancia(l3, l1), covariancia(l3, l2), variancia(l3)]
])

print(Sigma)
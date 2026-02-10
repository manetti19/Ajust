import random
import math

r = 1.0

A_quad = (2 * r) * (2 * r)

M = 10000000
N_dentro = 0

for _ in range(M):
    x = random.uniform(-r, r)
    y = random.uniform(-r, r)

    if x**2 + y**2 <= r**2:
        N_dentro += 1

frac = N_dentro / M

A_circ_est = A_quad * frac

A_circ_exata = math.pi * r**2

print(f"Raio r = {r}")
print(f"Área do quadrado      = {A_quad:.6f}")
print(f"Pontos dentro do círculo = {N_dentro} de {M}")
print(f"Área estimada (Monte Carlo) = {A_circ_est:.6f}")
print(f"Área exata                 = {A_circ_exata:.6f}")
print(f"Erro relativo              = {abs(A_circ_est - A_circ_exata)/A_circ_exata:.6%}")

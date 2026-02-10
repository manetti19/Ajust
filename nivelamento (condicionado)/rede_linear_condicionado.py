import numpy as np

# ---------------------------
# Observações medidas (desníveis)
# ---------------------------
Lb = np.array([+6.16, +12.57, +6.41, +1.09, +11.58, +5.07], float)

# ---------------------------
# Pesos inversamente proporcionais ao comprimento
# d = [4,2,2,4,2,4] km
# sigma^2 = d  →  peso = 1/d
# ---------------------------
d = np.array([4,2,2,4,2,4], float)
P = np.diag(1/d)
Pinv = np.diag(d)

# ---------------------------
# Matrizes de condição B (3 equações × 6 observações)
# ---------------------------
B = np.array([
    [ 1,  -1,  1, 0,  0,  0],   # F1 = l1 + l3 - l2
    [ 0,  0,  1,  0, -1,  1],   # F2 = l6 + l3 - l5
    [ 0,  -1,  0,  1, 1,  0]    # F3 = l4 + l5 - l2
], float)

# ---------------------------
# W = F(Lb)
# ---------------------------
def F(L):
    return np.array([
        L[0] + L[2] - L[1],
        L[5] + L[2] - L[4],
        L[3] + L[4] - L[1]
    ])

W = F(Lb)

# ---------------------------
# M = B * Pinv * B^T
# ---------------------------
M = B @ Pinv @ B.T

# ---------------------------
# K = - M^{-1} W
# ---------------------------
K = - np.linalg.solve(M, W)

# ---------------------------
# V = Pinv * B^T * K
# ---------------------------
V = Pinv @ (B.T @ K)

# ---------------------------
# L ajustado
# ---------------------------
La = Lb + V

print("\nDesníveis ajustados:")
for i,val in enumerate(La,1):
    print(f"l{i} = {val:.4f} m")

# ---------------------------
# Altitudes ajustadas
# ---------------------------
H_A = 0.0
H_B = H_A + La[0]
H_C = H_A + La[1]
H_D = H_A + La[3]

print("\nAltitudes ajustadas:")
print(f"H_A = {H_A:.4f} m")
print(f"H_B = {H_B:.4f} m")
print(f"H_C = {H_C:.4f} m")
print(f"H_D = {H_D:.4f} m")
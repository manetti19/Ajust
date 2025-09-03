from math import sqrt
import numpy as np

def sigma_C_from_A_B(sigma_A_arcsec: float, sigma_B_arcsec: float, cov_AB_arcsec2: float = 0.0):
    J = np.array([[-1.0, -1.0]])
    Sigma_L = np.array([[sigma_A_arcsec**2, cov_AB_arcsec2],
                        [cov_AB_arcsec2, sigma_B_arcsec**2]], dtype=float)
    Sigma_C = J @ Sigma_L @ J.T
    sigma_C_arcsec = float(np.sqrt(Sigma_C[0,0]))
    return sigma_C_arcsec


sigma_A = sqrt(3.0)  # √3 arcsec
sigma_B = sqrt(4.0)  # √4 = 2 arcsec
cov_AB = 0.0         # ajuste se houver correlação: cov_AB = ρ * sigma_A * sigma_B
sigma_C = sigma_C_from_A_B(sigma_A, sigma_B, cov_AB)
print(f"sigma_C = {sigma_C:.6f} arcsec")  # ≈ √7 ≈ 2.645751 arcsec

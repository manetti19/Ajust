p = print
import math 
import numpy as np
from numpy.linalg import inv, det, eig, svd, norm, solve

A = np.array([[5, -7],
              [3, 1],
              [1, -9],
              [3, -1],
              [6, -1]])

L = np.array([[-14],
             [8],
             [-23],
             [10],
             [6]])

AT = A.T
x_norm = np.linalg.solve(AT @ A, AT @ L)  # melhor que inv(AT@A) @ AT @ L
print("x (normais) =", x_norm)
B = A @ x_norm
p(B)
C = L - B
p(C)
a = (C[0,0] + C[1,0] + C[2,0] + C[3,0] + C[4,0])/5
Q = np.ones((5, 1))
E = abs(C)
p(E)
D = a*Q - E
p(D)
dp = (((D[0,0]- a)**(2) + (D[1,0]- a)**(2) + (D[2,0]- a)**(2) + (D[3,0]- a)**(2) + (D[4,0]- a)**(2))/5)**(1/2)
p(dp)
p(a + dp)






A = np.array([[5, -7],
              [3, 1],
              [1, -9],
              [6, -1]])

L = np.array([[-14],
             [8],
             [-23],
             [6]])

AT = A.T
x_norm = np.linalg.solve(AT @ A, AT @ L)  # melhor que inv(AT@A) @ AT @ L
print("x (normais) =", x_norm)
B = A @ x_norm
p(B)
C = L - B
p(C)
a = (C[0,0] + C[1,0] + C[2,0] + C[3,0])/4
Q = np.ones((4, 1))
E = abs(C)
p(E)
D = a*Q - E
p(D)
dp = (((D[0,0]- a)**(2) + (D[1,0]- a)**(2) + (D[2,0]- a)**(2) + (D[3,0]- a)**(2))/5)**(1/2)
p(dp)
p(a + dp)
p("resposta final")
p(x_norm)

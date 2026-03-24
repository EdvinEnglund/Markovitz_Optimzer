import numpy as np
import math
#Covariance matrix
C = np.array([[4.01,-1.19,0.60,0.74,-0.21],
              [-1.19,1.12,0.21,-0.54,0.55],
              [0.60,0.21,3.31,0.77,0.29],
              [0.74,-0.54,0.77,3.74,-1.04],
              [-0.21,0.55,0.29,-1.04,2.6]]) * 1e-2

n = C.shape[0] # n = 5

# form block matrix
A = np.block([[2*C, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1,1))]])

b = np.concatenate([np.zeros(n),np.array([1.0])])

# solve for x = weights, lambda
x = np.linalg.solve(A,b)

w = x[:n]
l = x[-1]

# calculate variance from objective function

var = w.T @ C @ w

print("optimal weights:")
for i in range(n):
    print(f"w{i}: {w[i]:.3f}")
print(f"variance: {var:.5f}")

r = np.array([0.13, 0.044, 0.121, 0.071, 0.117])
#for the expected return, we simply do the dot product of weights and the expected returns for the five assets
print(f"expected return of portfolio of least variance: {np.dot(r, w):.5f}")

#########################
##                     ##
## Irving Gomez Mendez ##
##  February 27, 2021  ##
##                     ##
#########################

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm
import random

beta_0 = 1
beta_1 = 1.5

x = np.linspace(0, 5, 20)
x = np.append(x,[6])
n = len(x)

np.random.seed(111)
y = beta_0+beta_1*x+norm.rvs(0,0.5,n)

S = x.var()*(n/(n-1))
D_M_2 = 1/S * np.diag((x-x.mean()).reshape(21,1) @ (x-x.mean()).reshape(21,1).T)

# influence of the points
h_ii = 1/n + D_M_2/(n-1)
h_ii

X = np.vstack([np.ones(n), x]).T
p_ii = np.diag(X @ np.linalg.solve(X.T @ X, X.T))
p_ii

plt.figure(figsize=(7.5,7.5))
plt.scatter(x, y, label='Original data', c=h_ii)
plt.colorbar()

# 11 points
x = np.linspace(0, 5, 10)
x = np.append(x,[6])
n = len(x)

np.random.seed(111)
y = beta_0+beta_1*x+norm.rvs(0,0.5,n)
X = np.vstack([np.ones(n), x]).T
p_ii = np.diag(X @ np.linalg.solve(X.T @ X, X.T))

plt.figure(figsize=(7.5,7.5))
plt.scatter(x, y, label='Original data', c=p_ii)
plt.colorbar()

# 6 points
x = np.linspace(0, 5, 5)
x = np.append(x,[6])
n = len(x)

np.random.seed(111)
y = beta_0+beta_1*x+norm.rvs(0,0.5,n)
X = np.vstack([np.ones(n), x]).T
p_ii = np.diag(X @ np.linalg.solve(X.T @ X, X.T))

plt.figure(figsize=(7.5,7.5))
plt.scatter(x, y, label='Original data', c=p_ii)
plt.colorbar()

# -*- coding: utf-8 -*-
"""
Created on Sat May 15 12:52:07 2021

@author: vdae_
"""

import numpy as np
import scipy.stats as stats
import scipy.optimize as optim
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

np.random.seed(0)

# Inliers
in_x = stats.uniform.rvs(-5, 10, size = 100)
in_y = stats.norm.rvs(3 + 5*in_x, 1)

# Outliers
out_x = stats.uniform.rvs(0, 5, size = 20)
out_y = stats.norm.rvs(-15 + out_x, 1)

xs = np.append(in_x, out_x).reshape(-1, 1)
ys = np.append(in_y, out_y)

###############################################################################
# Mínimos cuadrados ordinarios.                                               #
###############################################################################
xrange = np.linspace(-5, 5)
OLS = lm.LinearRegression(fit_intercept = True).fit(xs, ys)

plt.figure(figsize = (8, 6))
plt.title('Mínimos cuadrados ordinarios')
plt.xlabel('Variable independiente')
plt.ylabel('Variable dependiente')
plt.scatter(xs, ys)
plt.plot(xrange, OLS.intercept_ + OLS.coef_*xrange, 'k-', lw = 2)
plt.show()

###############################################################################
# Mínimas desviaciones absolutas.                                             #
###############################################################################
max_iter = 100
weights = np.ones(len(xs))
for i in range(max_iter):
    LAD = lm.LinearRegression(fit_intercept = True).fit(xs, ys, weights)
    weights = 1/np.maximum(1e-4, np.abs(ys - LAD.predict(xs)))
    
plt.figure(figsize = (8, 6))
plt.title('Mínimas desviaciones absolutas')
plt.xlabel('Variable independiente')
plt.ylabel('Variable dependiente')
plt.scatter(xs, ys, c = 1/weights, cmap = 'coolwarm')
plt.plot(xrange, LAD.intercept_ + LAD.coef_*xrange, 'k-', lw = 2)
plt.colorbar().set_label('Residuos')
plt.show()

###############################################################################
# Regresión de Huber.                                                         #
###############################################################################
Huber = lm.HuberRegressor(epsilon = 1.5, alpha = 0, fit_intercept = True).fit(xs, ys)

plt.figure(figsize = (8, 6))
plt.title('Regresión de Huber')
plt.xlabel('Variable independiente')
plt.ylabel('Variable dependiente')
plt.scatter(xs, ys)
plt.plot(xrange, Huber.intercept_ + Huber.coef_*xrange, 'k-', lw = 2)
plt.show()

###############################################################################
# Mínima mediana de cuadrados.                                                #
###############################################################################
def lms_loss(beta):
    res = ys - beta[0] - beta[1]*xs.flatten()
    return np.median(res**2)

LMS = optim.minimize(lms_loss, [0, 1]).x

plt.figure(figsize = (8, 6))
plt.title('Mínima mediana de cuadrados')
plt.xlabel('Variable independiente')
plt.ylabel('Variable dependiente')
plt.scatter(xs, ys)
plt.plot(xrange, LMS[0] + LMS[1]*xrange, 'k-', lw = 2)
plt.show()

###############################################################################
# Mínimos cuadrados truncados.                                                #
###############################################################################
def lts_loss(beta):
    res = ys - beta[0] - beta[1]*xs.flatten()
    return stats.trim_mean(res**2, 0.25)

LTS = optim.minimize(lts_loss, [0, 1]).x

plt.figure(figsize = (8, 6))
plt.title('Mínimos cuadrados truncados')
plt.xlabel('Variable independiente')
plt.ylabel('Variable dependiente')
plt.scatter(xs, ys)
plt.plot(xrange, LTS[0] + LTS[1]*xrange, 'k-', lw = 2)
plt.show()

###############################################################################
# RANSAC.                                                                     #
###############################################################################
max_iter = 100
tol = 10
thresh = 90

for i in range(max_iter):
    sample = stats.randint.rvs(0, len(xs), size = 2)
    RANSAC = lm.LinearRegression(fit_intercept = True).fit(xs[sample], ys[sample])
    consensus = np.abs(ys - RANSAC.predict(xs)) < tol
    if np.sum(consensus) > thresh:
        RANSAC = lm.LinearRegression(fit_intercept = True).fit(xs[consensus], ys[consensus])
        break
    
plt.figure(figsize = (8, 6))
plt.title('Random Sample Consensus (RANSAC)')
plt.xlabel('Variable independiente')
plt.ylabel('Variable dependiente')
plt.scatter(xs[consensus], ys[consensus], label = 'Inliers')
plt.scatter(xs[~consensus], ys[~consensus], label = 'Outliers')
plt.plot(xrange, RANSAC.intercept_ + RANSAC.coef_*xrange, 'k-', lw = 2)
plt.legend()
plt.show()
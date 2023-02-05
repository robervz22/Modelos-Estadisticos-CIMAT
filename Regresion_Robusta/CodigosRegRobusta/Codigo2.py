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

# Outliers en las variables explicativas
out_xx=stats.uniform.rvs(20,30,size=5)
out_yy=stats.norm.rvs(5 + out_xx, 1)

out_x=np.append(out_x,out_xx)
out_y=np.append(out_y,out_yy)

xs = np.append(in_x, out_x).reshape(-1, 1)
ys = np.append(in_y, out_y)

###############################################################################
# Mínimos cuadrados ordinarios.                                               #
###############################################################################
xrange = np.linspace(-5,45)
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

################################################################################
# GM-Estimador Huber                                                           #
################################################################################

# Derivada de Huber

def psi(x,epsilon):
    if x<-epsilon:
        return -epsilon
    if -epsilon<=x and x<=epsilon: 
        return x
    if x>epsilon:
        return epsilon

# Matriz Sombrero y DFFITS
N=len(xs)
xs_=np.column_stack([np.ones(N),xs])
H=xs_@ np.linalg.inv(xs_.T @ xs_)@ xs_.T
h_ii=np.diag(H)
pi_i=np.sqrt((1-h_ii)/h_ii)

# Condicion Inicial (Alto punto de quiebre)

beta_0=np.array([LTS[0],LTS[1]])

# Escala Robusta

def robust_s(beta):
    error=ys - beta[0] - beta[1]*xs.flatten()
    value=np.median(abs(error-np.median(error)))/0.6745
    return value

s0=robust_s(beta_0)

# Matriz de pesos

def W(s,beta):
    w=np.ones(N)
    error=ys - beta[0] - beta[1]*xs.flatten()
    for i in range(N):
        aux=error[i]
        w[i]=psi(aux,1.5)/(aux/pi_i[i]*s)
    W_value=np.diag(w)
    return W_value

tol=1e-6

# Primer iteracion
beta_aux=beta_0
WW=W(s0,beta_aux)
beta_GM=np.linalg.inv(xs_.T@ WW @ xs_)@xs_.T@WW@ys

# Procedimiento Iterativo
j=0
while np.linalg.norm(beta_GM-beta_aux)>tol and j<max_iter:
    beta_aux=beta_GM
    s=robust_s(beta_aux)
    WW=W(s,beta_aux)
    beta_GM=np.linalg.inv(xs_.T@ WW @ xs_)@xs_.T@WW@ys
    j=j+1

# Gráficas 

fig, (ax1,ax2)=plt.subplots(1,2)

# Figura 1
ax1.set_title('Regresión de Huber')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.scatter(xs, ys)
ax1.plot(xrange, Huber.intercept_ + Huber.coef_*xrange, 'k-', lw = 2)


# Figura 2
ax2.set_title('GM Estimador-Huber')
ax2.scatter(xs, ys)
ax2.plot(xrange, beta_GM[0] + beta_GM[1]*xrange, 'k-', lw = 2)

plt.savefig('GM_Estimador.png')
plt.show()





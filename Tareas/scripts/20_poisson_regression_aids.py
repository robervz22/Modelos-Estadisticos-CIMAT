#########################
##                     ##
## Irving Gomez Mendez ##
##     May 05, 2021    ##
##                     ##
#########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm, poisson

# This example considers the number of cases of AIDS in Belgium (data from the 80's).
# We assume that a Poisson model is razonable (See book of Wood S.S. (2006) Generalized Additive Models).
cases = [12,14,33,50,67,74,123,141,165,204,253,246,240]
x = np.array(range(13))
years = x+1981

# significance level
alpha = 0.05

plt.figure(figsize=(10,7.5))
plt.plot(years, cases, 'o')
plt.xlabel('year')
plt.ylabel('cases')
plt.title("Cases of AIDS in Belgium")

n = len(cases)
X = np.vstack([np.ones(n), x]).T

# Poisson regression by hand
b = [3,0.2] # initial values

tolm   = 1e-6       # tolerance (minimum norm of the difference of the betas)
iterm  = 100        # maximum number of iterations
tolera = 1          # initialize tolera
itera  = 0          # initialize ittera
histo  = b          # initialize beta upgrade

while((tolera > tolm) and (itera < iterm)):
  lam    = np.exp(X @ b)
  D      = np.diag(lam)
  var_b  = np.linalg.inv(X.T @ D @ X)
  delta  = var_b @ X.T @ (cases-lam)
  b      = b+delta
  tolera = np.sqrt(sum(delta**2))
  histo  = np.vstack([histo, b])
  itera  = itera+1

histo

# Let's get the significance of the estimators
se_b = np.sqrt(np.diag(var_b))
z_score = b/se_b
p_value = 1-norm.cdf(np.abs(z_score))

z_score
p_value

plt.figure(figsize=(10,7.5))
plt.plot(years, cases, 'o', label='data')
plt.plot(years, lam, 'o', label='predicted')
plt.xlabel('year')
plt.ylabel('cases')
plt.title("Cases of AIDS in Belgium")
plt.legend(loc='best')

# Let's plot using the log(cases)
plt.figure(figsize=(10,7.5))
plt.plot(years, np.log(cases), 'o', label='data')
plt.plot(years, X @ b, 'o', label='estimated')
plt.xlabel('year')
plt.ylabel('cases')
plt.title("log of cases of AIDS in Belgium")

# It looks like there is a lack of structre, lets consider a quadratic fit
X = np.vstack([np.ones(n), x, x**2]).T

# Poisson regression by hand
b = [3,0.2,0.2] # initial values

tolm   = 1e-6       # tolerance (minimum norm of the difference of the betas)
iterm  = 100        # maximum number of iterations
tolera = 1          # initialize tolera
itera  = 0          # initialize ittera
histo  = b          # initialize beta upgrade

while((tolera > tolm) and (itera < iterm)):
  lam    = np.exp(X @ b)
  D      = np.diag(lam)
  var_b  = np.linalg.inv(X.T @ D @ X)
  delta  = var_b @ X.T @ (cases-lam)
  b      = b+delta
  tolera = np.sqrt(sum(delta**2))
  histo  = np.vstack([histo, b])
  itera  = itera+1

# Let's get the significance of the estimators
se_b = np.sqrt(np.diag(var_b))
z_score = b/se_b
p_value = 1-norm.cdf(np.abs(z_score))

plt.figure(figsize=(10,7.5))
plt.plot(years, cases, 'o', label='data')
plt.plot(years, lam, 'o', label='predicted')
plt.xlabel('year')
plt.ylabel('cases')
plt.title("Cases of AIDS in Belgium")
plt.legend(loc='best')

# Let's plot using the log(cases)
plt.figure(figsize=(10,7.5))
plt.plot(years, np.log(cases), 'o', label='data')
plt.plot(years, X @ b, 'o', label='predicted')
plt.xlabel('year')
plt.ylabel('cases')
plt.title("log of cases of AIDS in Belgium")
plt.legend(loc='upper left')

# Let's add confidence and prediction intervals
upp_conf = X @ b + norm.ppf(1-alpha/2)*np.sqrt(np.diag(X @ var_b @ X.T))
low_conf = X @ b - norm.ppf(1-alpha/2)*np.sqrt(np.diag(X @ var_b @ X.T))

# Let's plot using the log(cases)
plt.figure(figsize=(10,7.5))
plt.fill_between(years, low_conf, upp_conf, facecolor='yellow', alpha=0.5, label=' Approx. confidence interval')
plt.plot(years, np.log(cases), 'o', label='data')
plt.plot(years, X @ b, 'o', label='estimated')
plt.xlabel('year')
plt.ylabel('cases')
plt.title("log of cases of AIDS in Belgium")
plt.legend(loc='upper left')

upp_pred = poisson.ppf(1-alpha/2, lam)
low_pred = poisson.ppf(alpha/2, lam)

plt.figure(figsize=(10,7.5))
plt.fill_between(years, low_pred, upp_pred, facecolor='green', alpha=0.5, label='Prediction interval')
plt.plot(years, cases, 'o', label='data')
plt.plot(years, lam, 'o', label='predicted')
plt.xlabel('year')
plt.ylabel('cases')
plt.title("Cases of AIDS in Belgium")
plt.legend(loc='upper left')

upp_conf_y_scale = lam + norm.ppf(1-alpha/2)*np.sqrt(np.exp(2*X @ b) * np.diag(X @ var_b @ X.T))
low_conf_y_scale = lam - norm.ppf(1-alpha/2)*np.sqrt(np.exp(2*X @ b) * np.diag(X @ var_b @ X.T))

# We can add the confidence interval, with the warning due to Jensen's inequality
plt.figure(figsize=(10,7.5))
plt.fill_between(years, low_pred, upp_pred, facecolor='green', alpha=0.5, label='Prediction interval')
plt.fill_between(years, low_conf_y_scale, upp_conf_y_scale, facecolor='yellow', alpha=0.5, label='Approx. confidence interval')
plt.plot(years, cases, 'o', label='data')
plt.plot(years, lam, 'o', label='predicted')
plt.xlabel('year')
plt.ylabel('cases')
plt.title("Cases of AIDS in Belgium")
plt.legend(loc='upper left')

# Using sm.GLM
poisson_model = sm.GLM(cases, X, family=sm.families.Poisson())
poisson_results = poisson_model.fit()
poisson_results.summary()

b
se_b
z_score



###

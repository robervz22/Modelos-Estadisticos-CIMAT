#########################
##                     ##
## Irving Gomez Mendez ##
##    March 07, 2021   ##
##                     ##
#########################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, beta, t

import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import OLSInfluence

#https://www.statsmodels.org/stable/stats.html

# Function to calculate the conficen interval for a qq plot
def QQ_norm_interval(n, k, alpha=0.05, mu=0, sigma=1):
    a1 = beta.ppf(alpha/2, k, n+1-k)
    a2 = beta.ppf(1-alpha/2, k, n+1-k)

    low_int = norm.ppf(a1, mu, sigma)
    upp_int = norm.ppf(a2, mu, sigma)

    return(low_int, upp_int)

# Data
dat = np.array([[3.00,  2.60, 1],
    [3.00,  2.67, 1],
    [3.00,  2.66, 1],
    [3.00,  2.78, 1],
    [3.00,  2.80, 1],
    [5.34,  5.92, 2],
    [5.38,  5.35, 2],
    [5.40,  4.33, 2],
    [5.40,  4.89, 2],
    [5.45,  5.21, 2],
    [7.70,  7.68, 3],
    [7.80,  9.81, 3],
    [7.81,  6.52, 3],
    [7.85,  9.71, 3],
    [7.87,  9.82, 3],
    [7.91,  9.81, 3],
    [7.94,  8.50, 3],
    [9.03,  9.47, 4],
    [9.07,  11.45, 4],
    [9.11,  12.14, 4],
    [9.14,  11.50, 4],
    [9.16,  10.65, 4],
    [9.37,  10.64, 4],
    [10.17,  9.78, 5],
    [10.18,  12.39, 5],
    [10.22,  11.03, 5],
    [10.22,  8.00, 5],
    [10.22,  11.90, 5],
    [10.18,  8.68, 5],
    [10.50,  7.25, 5],
    [10.23,  13.46, 5],
    [10.03,  10.19, 5],
    [10.23,  9.93, 5]])

dat = pd.DataFrame(dat,columns=['x', 'y', 'grp'])
n = dat.shape[0]

dat

# Get the confidence intervals for the qq plot
kk = np.linspace(1,n,n)
qq_low_intervals, qq_upp_intervals = QQ_norm_interval(n, kk)

# Design matrix
XX = sm.add_constant(dat)

# inciso a)
# Ordinary least squares
results = sm.OLS(dat['y'],XX[['const','x']]).fit()

# Get confidence and prediction intervals
x0 = np.linspace(XX['x'].min(),XX['x'].max(),50)
XX0 = sm.add_constant(x0)
predictions = results.get_prediction(XX0)
predictions_info = predictions.summary_frame(alpha=0.05)

# Graph
plt.figure(figsize=(10,5))
plt.fill_between(x0, predictions_info['obs_ci_lower'],
    predictions_info['obs_ci_upper'],
    facecolor='green', alpha=0.5, label='Prediction interval')
plt.fill_between(x0, predictions_info['mean_ci_lower'],
    predictions_info['mean_ci_upper'],
    facecolor='yellow', alpha=0.5, label='Confidence interval')
plt.scatter(XX['x'], dat['y'], label='Original data')
plt.scatter(XX['x'], results.fittedvalues, label='Fitted Values')
plt.title("Independent Variable X Target Variable")
plt.xlabel("Independent Variable")
plt.ylabel("Target Variable")
plt.legend(loc='upper left')

# inciso b)
plt.figure(figsize=(10,5))
plt.scatter(results.fittedvalues, results.resid)
plt.title("Residuals X Fitted Values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")

# inciso c)
plt.figure(figsize=(10,5))
plt.scatter(XX['x'], results.resid)
plt.title("Residuals X Independent Variable")
plt.xlabel("Independent Variable")
plt.ylabel("Residuals")

results.summary2()

std_resid = OLSInfluence(results).summary_frame()['standard_resid']

OLSInfluence(results).summary_frame()

h_ii = OLSInfluence(results).summary_frame()['hat_diag']

standard_residuals = results.resid/(np.sqrt(results.scale*(1-h_ii)))

SSE = sum(results.resid**2)
s2_without_i = (SSE-results.resid**2/(1-h_ii))/(30)
studentized_residuals = results.resid/(np.sqrt(s2_without_i*(1-h_ii)))

plt.scatter(results.resid,results.resid/(1-h_ii))

plt.plot(results.resid, 'o')

plt.plot(results.resid[1:],results.resid[:-1], 'o')

h_ii[h_ii > 4/n]

D_cook = OLSInfluence(results).summary_frame()['cooks_d']
D_cook[D_cook > 0.8]
D_cook[D_cook > 4/n]

DFFITS = results.resid/(np.sqrt(s2_without_i*h_ii))*(h_ii/(1-h_ii))
DFFITS_internal = results.resid/(np.sqrt(results.scale*h_ii))*(h_ii/(1-h_ii))

DFFITS[DFFITS > 2*np.sqrt(2/n)]

DF_beta0 = OLSInfluence(results).summary_frame()['dfb_const']
DF_beta1 = OLSInfluence(results).summary_frame()['dfb_x']

DF_beta0[DF_beta0>2/np.sqrt(n)]
DF_beta1[DF_beta1>2/np.sqrt(n)]

PRESS = sum((results.resid/(1-h_ii))**2)

SSE
PRESS

plt.figure(figsize=(10,5))
plt.hist(std_resid, density=True)
ll = np.linspace(std_resid.min(),std_resid.max(),20)
norm_density = norm.pdf(ll,std_resid.mean(),std_resid.std())
plt.plot(ll, norm_density, 'k', linewidth=2)
plt.title("Histogram and Normal Density for Standard Residuals of Ordinary Least Squares")

fig, ax = plt.subplots(figsize=(10,5))
fig=qqplot(std_resid, line='45', ax=ax)
plt.plot(norm.ppf(kk/(n+1)), qq_low_intervals, 'r--')
plt.plot(norm.ppf(kk/(n+1)), qq_upp_intervals, 'r--')
plt.title("QQ plot for Standard Residuals of the Ordinary Least Squares")

# Get the unbiased estimate of variance per group
dat_variance = dat.groupby('grp').var().reset_index()[['grp','y']]
dat_variance = dat_variance.rename(columns={'y':'variance'})
dat = dat.merge(dat_variance, how='left', on='grp')

# inciso d)
plt.figure(figsize=(10,5))
plt.scatter(XX['x'], dat['variance'])
plt.title("Variance per Group X Independent Variable")
plt.xlabel("Independent Variable")
plt.ylabel("Variance per Group")

# How is the relation between x and variance(y)?
aux_dat = dat[['x', 'variance']]
aux_dat = aux_dat.sort_values('x')
aux_dat = sm.add_constant(aux_dat)
aux_dat['x_2'] = aux_dat['x']**2
aux_dat['x_3'] = aux_dat['x']**3

# Fit the linear model
linear_fit = sm.OLS(aux_dat['variance'],aux_dat[['const','x']]).fit()
linear_fit_hat_y = linear_fit.predict(aux_dat0[['const','x']])
sum(linear_fit_hat_y < 0)

linear_fit_2 = sm.OLS(aux_dat['variance'],aux_dat[['x']]).fit()
linear_fit_2_hat_y = linear_fit_2.predict(aux_dat0[['x']])
sum(linear_fit_2_hat_y < 0)

linear_fit_2.summary2()

# Fit the quadratic model
quadratic_fit = sm.OLS(aux_dat['variance'],aux_dat[['const','x','x_2']]).fit()
quadratic_fit_hat_y = quadratic_fit.predict(aux_dat0[['const', 'x','x_2']])
sum(quadratic_fit_hat_y < 0)

quadratic_fit_2 = sm.OLS(aux_dat['variance'],aux_dat[['x','x_2']]).fit()
quadratic_fit_2_hat_y = quadratic_fit_2.predict(aux_dat0[['x','x_2']])
sum(quadratic_fit_2_hat_y < 0)

quadratic_fit_3 = sm.OLS(aux_dat['variance'],aux_dat[['const','x_2']]).fit()
quadratic_fit_3_hat_y = quadratic_fit_3.predict(aux_dat0[['const','x_2']])
sum(quadratic_fit_3_hat_y < 0)

quadratic_fit_4 = sm.OLS(aux_dat['variance'],aux_dat[['x_2']]).fit()
quadratic_fit_4_hat_y = quadratic_fit_4.predict(aux_dat0[['x_2']])
sum(quadratic_fit_4_hat_y < 0)

quadratic_fit_4.summary2()

# Fit the cubic model
cubic_fit = sm.OLS(aux_dat['variance'],aux_dat[['const','x','x_2','x_3']]).fit()
cubic_fit_hat_y = cubic_fit.predict(aux_dat0[['const','x','x_2','x_3']])
sum(cubic_fit_hat_y < 0)

cubic_fit_2 = sm.OLS(aux_dat['variance'],aux_dat[['x','x_2','x_3']]).fit()
cubic_fit_2_hat_y = cubic_fit_2.predict(aux_dat0[['x','x_2','x_3']])
sum(cubic_fit_2_hat_y < 0)

cubic_fit_3 = sm.OLS(aux_dat['variance'],aux_dat[['const','x_2','x_3']]).fit()
cubic_fit_3_hat_y = cubic_fit_3.predict(aux_dat0[['const','x_2','x_3']])
sum(cubic_fit_3_hat_y < 0)

cubic_fit_4 = sm.OLS(aux_dat['variance'],aux_dat[['const','x','x_3']]).fit()
cubic_fit_4_hat_y = cubic_fit_4.predict(aux_dat0[['const','x','x_3']])
sum(cubic_fit_4_hat_y < 0)

aux_dat0 = pd.DataFrame(np.linspace(XX['x'].min(),XX['x'].max(),50), columns=['x'])
aux_dat0 = sm.add_constant(aux_dat0)
aux_dat0['x_2'] = aux_dat0['x']**2
aux_dat0['x_3'] = aux_dat0['x']**3

# Graph with the three models together
plt.figure(figsize=(10,5))
plt.plot(x0, linear_fit_2_hat_y, label='b*x')
plt.plot(x0, quadratic_fit_4_hat_y, label='c*x**2')
plt.plot(x0, cubic_fit_2_hat_y, label='b*x+c*x**2+d*x**3')
plt.plot(x0, cubic_fit_3_hat_y, label='a+c*x**2+d*x**3')
plt.plot(x0, cubic_fit_4_hat_y, label='a+b*x+d*x**3')
plt.scatter(XX['x'], dat['variance'], label='Data')
plt.title("Variance per Group X Independent Variable")
plt.xlabel("Independent Variable")
plt.ylabel("Variance per Group")
plt.legend(loc='upper left')

cubic_fit_2.summary2()

cubic_fit_3.summary2()

cubic_fit_4.summary2()

cubic_fit_5 = sm.OLS(aux_dat['variance'],aux_dat[['x_2','x_3']]).fit()
cubic_fit_5_hat_y = cubic_fit_5.predict(aux_dat0[['x_2','x_3']])
sum(cubic_fit_5_hat_y < 0)

cubic_fit_6 = sm.OLS(aux_dat['variance'],aux_dat[['const','x_3']]).fit()
cubic_fit_6_hat_y = cubic_fit_6.predict(aux_dat0[['const','x_3']])
sum(cubic_fit_6_hat_y < 0)

cubic_fit_7 = sm.OLS(aux_dat['variance'],aux_dat[['x','x_3']]).fit()
cubic_fit_7_hat_y = cubic_fit_7.predict(aux_dat0[['x','x_3']])
sum(cubic_fit_7_hat_y < 0)

cubic_fit_8 = sm.OLS(aux_dat['variance'],aux_dat[['x_3']]).fit()
cubic_fit_8_hat_y = cubic_fit_8.predict(aux_dat0[['x_3']])
sum(cubic_fit_8_hat_y < 0)

cubic_fit_8.summary2()

# Thus, we select the models x, x**2 and x**3

plt.figure(figsize=(10,5))
plt.plot(x0, linear_fit_2_hat_y, label='Linear Fit')
plt.plot(x0, quadratic_fit_4_hat_y, label='Quadratic Fit')
plt.plot(x0, cubic_fit_8_hat_y, label='Cubic Fit')
plt.scatter(XX['x'], dat['variance'], label='Data')
plt.title("Variance per Group X Independent Variable")
plt.xlabel("Independent Variable")
plt.ylabel("Variance per Group")
plt.legend(loc='upper left')

##################
##              ##
##  Linear Fit  ##
##              ##
##################
predictions_linear_fit_2 = linear_fit_2.get_prediction(aux_dat0[['x']])
predictions_linear_fit_2_info = predictions_linear_fit_2.summary_frame(alpha=0.05)

plt.figure(figsize=(10,5))
plt.fill_between(x0, predictions_linear_fit_2_info['obs_ci_lower'],
    predictions_quadratic_fit_4_info['obs_ci_upper'],
    facecolor='green', alpha=0.5, label='Prediction interval')
plt.fill_between(x0, predictions_linear_fit_2_info['mean_ci_lower'],
    predictions_linear_fit_2_info['mean_ci_upper'],
    facecolor='yellow', alpha=0.5, label='Confidence interval')
plt.plot(x0, linear_fit_2.predict(aux_dat0[['x']]), label='Fitted Line')
plt.scatter(XX['x'], dat['variance'], label='Original data')
plt.title("Variance per Group X Independent Variable")
plt.xlabel("Independent Variable")
plt.ylabel("Variance per Group")
plt.legend(loc='upper left')

plt.figure(figsize=(10,5))
plt.scatter(XX['x'], linear_fit_2.resid)
plt.title("Residuals for Quadratic Model X Independent Variable")
plt.xlabel("Independent Variable")
plt.ylabel("Residuals for Quadratic Model")

linear_std_resid = OLSInfluence(linear_fit_2).summary_frame()['standard_resid']

plt.figure(figsize=(10,5))
plt.hist(linear_std_resid, density=True)
ll = np.linspace(linear_std_resid.min(),linear_std_resid.max(),20)
norm_density = norm.pdf(ll,linear_std_resid.mean(),linear_std_resid.std())
plt.plot(ll, norm_density, 'k', linewidth=2)
plt.title("Histogram and Normal Density for Standard Residuals of the Linear Model")

fig, ax = plt.subplots(figsize=(10,5))
fig=qqplot(linear_std_resid, line='45', ax=ax)
plt.plot(norm.ppf(kk/(n+1)), qq_low_intervals, 'r--')
plt.plot(norm.ppf(kk/(n+1)), qq_upp_intervals, 'r--')
plt.title("QQ plot for Standard Residuals of the Linear Model")

#####################
##                 ##
##  Quadratic Fit  ##
##                 ##
#####################
predictions_quadratic_fit_4 = quadratic_fit_4.get_prediction(aux_dat0[['x_2']])
predictions_quadratic_fit_4_info = predictions_quadratic_fit_4.summary_frame(alpha=0.05)

plt.figure(figsize=(10,5))
plt.fill_between(x0, predictions_quadratic_fit_4_info['obs_ci_lower'],
    predictions_quadratic_fit_4_info['obs_ci_upper'],
    facecolor='green', alpha=0.5, label='Prediction interval')
plt.fill_between(x0, predictions_quadratic_fit_4_info['mean_ci_lower'],
    predictions_quadratic_fit_4_info['mean_ci_upper'],
    facecolor='yellow', alpha=0.5, label='Confidence interval')
plt.plot(x0, quadratic_fit_4.predict(aux_dat0[['x_2']]), label='Fitted Line')
plt.scatter(XX['x'], dat['variance'], label='Original data')
plt.title("Variance per Group X Independent Variable")
plt.xlabel("Independent Variable")
plt.ylabel("Variance per Group")
plt.legend(loc='upper left')

plt.figure(figsize=(10,5))
plt.scatter(XX['x'], quadratic_fit_4.resid)
plt.title("Residuals for Quadratic Model X Independent Variable")
plt.xlabel("Independent Variable")
plt.ylabel("Residuals for Quadratic Model")

quadratic_std_resid = OLSInfluence(quadratic_fit_4).summary_frame()['standard_resid']

plt.figure(figsize=(10,5))
plt.hist(quadratic_std_resid, density=True)
ll = np.linspace(quadratic_std_resid.min(),quadratic_std_resid.max(),20)
norm_density = norm.pdf(ll,quadratic_std_resid.mean(), quadratic_std_resid.std())
plt.plot(ll, norm_density, 'k', linewidth=2)
plt.title("Histogram and Normal Density for Standard Residuals of the Quadratic Model")

fig, ax = plt.subplots(figsize=(10,5))
fig=qqplot(quadratic_std_resid, line='45', ax=ax)
plt.plot(norm.ppf(kk/(n+1)), qq_low_intervals, 'r--')
plt.plot(norm.ppf(kk/(n+1)), qq_upp_intervals, 'r--')
plt.title("QQ plot for Standard Residuals of the Quadratic Model")

#################
##             ##
##  Cubic Fit  ##
##             ##
#################
predictions_cubic_fit_8 = cubic_fit_8.get_prediction(aux_dat0[['x_3']])
predictions_cubic_fit_8_info = predictions_cubic_fit_8.summary_frame(alpha=0.05)

plt.figure(figsize=(10,5))
plt.fill_between(x0, predictions_cubic_fit_8_info['obs_ci_lower'],
    predictions_cubic_fit_8_info['obs_ci_upper'],
    facecolor='green', alpha=0.5, label='Prediction interval')
plt.fill_between(x0, predictions_cubic_fit_8_info['mean_ci_lower'],
    predictions_cubic_fit_8_info['mean_ci_upper'],
    facecolor='yellow', alpha=0.5, label='Confidence interval')
plt.plot(x0, cubic_fit_8.predict(aux_dat0[['x_3']]), label='Fitted Line')
plt.scatter(XX['x'], dat['variance'], label='Original data')
plt.title("Variance per Group X Independent Variable")
plt.xlabel("Independent Variable")
plt.ylabel("Variance per Group")
plt.legend(loc='upper left')

plt.figure(figsize=(10,5))
plt.scatter(XX['x'], cubic_fit_8.resid)
plt.title("Residuals for Cubic Model X Independent Variable")
plt.xlabel("Independent Variable")
plt.ylabel("Residuals for Cubic Model")

cubic_std_resid = OLSInfluence(cubic_fit_8).summary_frame()['standard_resid']

plt.figure(figsize=(10,5))
plt.hist(cubic_std_resid, density=True)
ll = np.linspace(cubic_std_resid.min(),cubic_std_resid.max(),20)
norm_density = norm.pdf(ll,cubic_std_resid.mean(), cubic_std_resid.std())
plt.plot(ll, norm_density, 'k', linewidth=2)
plt.title("Histogram and Normal Density for Standard Residuals of the Cubic Model")

fig, ax = plt.subplots(figsize=(10,5))
fig=qqplot(cubic_std_resid, line='45', ax=ax)
plt.plot(norm.ppf(kk/(n+1)), qq_low_intervals, 'r--')
plt.plot(norm.ppf(kk/(n+1)), qq_upp_intervals, 'r--')
plt.title("QQ plot for Standard Residuals of the Cubic Model")

# inciso e)
dat['weight'] = 1/dat['variance']
XX['weight'] = dat['weight']

# inciso f)
# Weighted least squares
results_weight = sm.WLS(dat['y'],XX[['const','x']],XX['weight']).fit()

# Necesary stuff to calculate (aprox.) a prediction interval
alpha = 0.05
predictions_weight = results_weight.get_prediction(XX0)
predictions_weight_info = predictions_weight.summary_frame(alpha=0.05)

y_hat = results_weight.predict(XX0)
hat_sigma_2 = results_weight.mse_resid

f2 = predictions_quadratic_fit_4_info['mean']
V = np.diag(f2)
f2 = np.array(f2).reshape(len(f2),1)
V_inv = np.linalg.inv(V)

int_sqrt_aux = np.diag(f2 + np.array(aux_dat0) @ np.linalg.inv(aux_dat0.T @ V_inv @ aux_dat0) @ np.array(aux_dat0).T)
aux_t_pred   = np.sqrt(hat_sigma_2 * int_sqrt_aux)
hat_y_low_pred = y_hat - t.ppf(1-alpha/2,n-2)*aux_t_pred
hat_y_upp_pred = y_hat + t.ppf(1-alpha/2,n-2)*aux_t_pred

# Graph
plt.figure(figsize=(10,5))
plt.fill_between(x0, hat_y_low_pred, hat_y_upp_pred,
    facecolor='green', alpha=0.5, label='Prediction interval')
plt.fill_between(x0, predictions_weight_info['mean_ci_lower'],
    predictions_weight_info['mean_ci_upper'],
    facecolor='yellow', alpha=0.5, label='Confidence interval')
plt.scatter(XX['x'], dat['y'], label='Original data')
plt.scatter(XX['x'], results_weight.fittedvalues, label='Fitted Values')
plt.title("Independent Variable X Target Variable")
plt.xlabel("Independent Variable")
plt.ylabel("Target Variable")
plt.legend(loc='upper left')

plt.figure(figsize=(10,5))
plt.scatter(results_weight.fittedvalues, np.sqrt(XX['weight'])*results_weight.resid)
plt.title("Residuals X Fitted Values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")

results_weight.summary2()

std_resid_weight = OLSInfluence(results_weight).summary_frame()['standard_resid']
std_resid_weight = np.sqrt(XX['weight'])*std_resid_weight

plt.figure(figsize=(10,5))
plt.hist(std_resid_weight, density=True)
ll = np.linspace(std_resid_weight.min(),std_resid_weight.max(),20)
norm_density = norm.pdf(ll,std_resid_weight.mean(),std_resid_weight.std())
plt.plot(ll, norm_density, 'k', linewidth=2)
plt.title("Histogram and Normal Density for Weighted Standard Residuals of Weighted Least Squares")

fig, ax = plt.subplots(figsize=(10,5))
fig=qqplot(std_resid_weight, line='45', ax=ax)
plt.plot(norm.ppf(kk/(n+1)), qq_low_intervals, 'r--')
plt.plot(norm.ppf(kk/(n+1)), qq_upp_intervals, 'r--')
plt.title("QQ plot for Weighted Standard Residuals of the Weighted Least Squares")

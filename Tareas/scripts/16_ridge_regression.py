#########################
##                     ##
## Irving Gomez Mendez ##
##  February 27, 2021  ##
##                     ##
#########################

import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error

data = pd.read_csv("prostate_dataset.txt", sep='\t')
data = data.drop(['col'], axis=1)

data
# The data were collected on n=97 men before radical prostatectomy,
# which a major surgical operation that removes the entire prostate
# gland along with some surrounding tissue.

# lcavol  = log cancer volumne, measure in milliliters (cc). The area
#           of cancer was measure from digitized images and multiplied
#           by a thickness to produce a volume.
# lweight = log prostate weight
# age     = age
# lbph    = log of the amount of benign prostatic hyperplasia, a
#           noncancerous enlargement of the prostate gland, as an area in
#           a digitized image and reported in cm2.
# svi     = seminal vesicle invasion, a 0/1 indicator of whether prostate
#           cancer cells hae invaded the vesicle.
# lcp     = log capsular penetration, which represents the level of
#           extension of cancer into the capsule (the fibrous tissue
#           which acts as an outer lining of the prostate gland).
#           Measure as the linear extent of penetration, in cm.
# gleason = Gleason score, a measure of the degree of aggressiveness of
#           the tumor. The Gleason grading system assigns a grade
#           to each of the two largest areas of cancer in the tissue
#           samples with 1 being the least aggressive and 5 the most
#           aggressive; the two grades are then added together to produce
#           the Gleason score.
# pgg45   = percent of Gleason score 4 or 5.
# lpsa    = log prostate specific antigen

# See Hastie et al (2008) The Elements of Stat. Learning (pp. 49)
# and Wakefiled (2013) (pp. 5)

# PSA is a concentration and is measure in ng/ml.
# In Stamey et al (1989), PSA was proposed as a preoperative marker
# to predict  the clinical stage of cancer.
# PSA is a protein produced by the cells of the prostate gland.
# PSA is present in small quantities in the serum of men with healthy
# prostates, but is often elevated in the presence of prostate cancer
# and in other prostate disorders. A blood test to measure PSA is
# considered the most effective test currently available for the early
# detection of prostate cancer, but this effectiveness has also
# been questioned. Rising levels of PSA over time are associated
# with both localized and metastatic prostate cancer.

#################################################
##                                             ##
## Creating the training and testing data sets ##
##                                             ##
#################################################

train_data = data[data['train'] == 'T']
test_data = data[data['train'] == 'F']

train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

data = data.drop(['train'], axis=1)
train_data = train_data.drop(['train'], axis=1)
test_data = test_data.drop(['train'], axis=1)

X_train = train_data.drop('lpsa', axis=1)
y_train = train_data['lpsa']

X_test = test_data.drop('lpsa', axis=1)
y_test = test_data['lpsa']

###############################
##                           ##
## Exploring multicolinearty ##
##                           ##
###############################

corr_matrix = train_data.corr()
plt.figure(figsize=(10,7.5))
sns.heatmap(corr_matrix, cmap='Purples')

sns.pairplot(train_data, kind="reg", corner = True,
    plot_kws={'line_kws':{'color':'orange'}, 'scatter_kws': {'color': 'rebeccapurple'}},
    diag_kws={'color': 'rebeccapurple'})

# Let's see the eigenvalues of the matrix XTX
u, d, vh = np.linalg.svd(X_train, full_matrices=True)
np.round(d**2,2)

n_lambdas = 200
lambdas = np.logspace(0, 5, n_lambdas)
df_l = []
coefs = []

for l in lambdas:
    df_l.append(sum(d**2/(d**2+l)))
    ridge = Ridge(alpha=l, fit_intercept=True).fit(X_train, y_train)
    coefs.append(ridge.coef_)

fig, ax = plt.subplots(figsize=(10,7.5))
ax.plot(df_l, coefs)
ax.set_xlabel('df')
ax.set_ylabel('betas')
ax.set_title('Ridge coefficients as a function of effective degrees of freedom')
ax.grid(True)
plt.legend(X_train.columns)

fig, ax = plt.subplots(figsize=(10,7.5))
ax.plot(lambdas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # decreasing time
ax.set_xlabel('lambda')
ax.set_ylabel('betas')
ax.set_title('Ridge coefficients as a function of the regularization')
ax.grid(True)
plt.legend(X_train.columns)

n_repeats = 1
n_splits = 67
kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)

MSE_per_lambda_and_K_fold = np.zeros((n_repeats*n_splits, n_lambdas))

for i, index in enumerate(kf.split(X_train)):
    train_index, valid_index = index
    X_Kfold, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_Kfold, y_valid = y_train[train_index], y_train[valid_index]

    for j, l in enumerate(lambdas):
        ridge = Ridge(alpha=l).fit(X_Kfold, y_Kfold)
        hat_y_valid = ridge.predict(X_valid)
        MSE_per_lambda_and_K_fold[i,j] = mean_squared_error(y_valid, hat_y_valid)

mean_MSE = MSE_per_lambda_and_K_fold.mean(axis=0)

plt.figure(figsize=(10,7.5))
plt.plot(df_l, mean_MSE)

hat_lambda = lambdas[np.argmin(mean_MSE)]
hat_lambda
sum(d**2/(d**2+hat_lambda))
mean_MSE.min()

model_ridge = Ridge(alpha=hat_lambda).fit(X_train,y_train)
model_ridge.coef_
model_ridge.intercept_

clf = RidgeCV(alphas=lambdas).fit(X_train, y_train)
clf.alpha_
clf.best_score_
clf.coef_
clf.intercept_

mean_squared_error(y_test, clf.predict(X_test))

fig, ax = plt.subplots(figsize=(10,7.5))
ax.plot(df_l, coefs)
ax.set_xlabel('df')
ax.set_ylabel('betas')
ax.set_title('Ridge coefficients as a function of effective degrees of freedom')
ax.grid(True)
plt.axvline(sum(d**2/(d**2+hat_lambda)), c='r', ls='--')
plt.legend(X_train.columns)

fig, ax = plt.subplots(figsize=(10,7.5))
ax.plot(lambdas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # decreasing time
ax.set_xlabel('lambda')
ax.set_ylabel('betas')
ax.set_title('Ridge coefficients as a function of the regularization')
ax.grid(True)
plt.axvline(clf.alpha_, c='r', ls='--')
plt.legend(X_train.columns)



####

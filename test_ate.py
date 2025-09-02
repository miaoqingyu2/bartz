import os

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from functools import partial

# pip install -e.
import jax
import pandas
from jax import numpy as jnp
from jax import random
from collections import namedtuple

from time import perf_counter
from bartz.BART import gbart
from sklearn.linear_model import LogisticRegression

outcomes = pandas.read_csv('./data/NHANES_outcomes.csv')
predictors = pandas.read_csv('./data/NHANES_predictors.csv')
outcomes_names = outcomes.columns.tolist()
X_full_raw = predictors.values
y_full_raw = outcomes.values

y_full = y_full_raw / jnp.std(y_full_raw, axis=0)
Z = X_full_raw[:, [0]]               # shape (n, 1) — treatment
X_covariates = X_full_raw[:, 1:4]    # use columns 2 to 4 

# 4. Split into train/test
key = random.key(42)
n_total = X_covariates.shape[0]
train_frac = 0.7
n_train = int(train_frac * n_total)

# Create random permutation
perm = random.permutation(key, n_total)
train_idx = perm[:n_train]
test_idx = perm[n_train:]

# Split X, Y, Z
X_train = X_covariates[train_idx, :]
X_test = X_covariates[test_idx, :]
y_train = y_full[train_idx, :]
y_test = y_full[test_idx, :]
Z_train = Z[train_idx, :].astype(jnp.float32)
Z_test = Z[test_idx, :].astype(jnp.float32)

ps_model = LogisticRegression(solver='liblinear', random_state=42)
ps_model.fit(X_train, Z_train)

# Get the predicted probabilities of treatment (Z=1)
# .predict_proba returns probabilities for [class 0, class 1]
# We need the second column for P(Z=1|X)
ps_train = ps_model.predict_proba(X_train)[:, 1]
ps_test = ps_model.predict_proba(X_test)[:, 1]

# 6. Append propensity score to covariates
X_train_ps = jnp.concatenate([X_train, ps_train[:, None]], axis=1)
X_test_ps = jnp.concatenate([X_test, ps_test[:, None]], axis=1)

print(f"Final covariate matrix shape (with ps): {X_train_ps.shape}")

# 7. Put into namedtuple (transposed for gbart)
Data = namedtuple('Data', 'X_train X_train_with_ps y_train X_test X_test_with_ps y_test Z_train Z_test ps_train ps_test')
data = Data(
    X_train=X_train.T,
    X_test=X_test.T,
    X_train_with_ps=X_train_ps.T,
    y_train=y_train,
    X_test_with_ps=X_test_ps.T,
    y_test=y_test,
    Z_train=Z_train,
    Z_test=Z_test,
    ps_train=ps_train,
    ps_test=ps_test
)

device = jax.devices()[0]
data = jax.device_put(data, device)

print("Data loaded, standardized, split, and sent to device.")



n_tree = 200  # number of trees used by bartz

keys = list(random.split(random.key(202404161853), 2))
# run bartz
start = perf_counter()
mvbart = gbart(
    data.X_train_with_ps,
    data.y_train,
    ntree=n_tree,
    nskip=1000,
    ndpost=1000,
    printevery=100,
    seed=keys.pop(),
)

end = perf_counter()

# compute predictions
yhat_test = mvbart.predict(data.X_test_with_ps)  # posterior samples, n_samples x n_test
yhat_test_mean = jnp.mean(yhat_test, axis=0)  # posterior mean point-by-point
yhat_test_var = jnp.var(yhat_test, axis=0)  # posterior variance point-by-point
sigam2_cov = mvbart.sigma2_cov_mean
sigma2_cov_prec = mvbart.sigma2_cov_prec_mean
print('yhat_test_mean is, ', yhat_test_mean)
print('covariance matrix inverse is, ', sigma2_cov_prec)

# consider ATE
X_test_treated = jnp.concatenate([data.X_test, jnp.ones((data.ps_test.shape[0], 1)).T], axis=0)
X_test_control = jnp.concatenate([data.X_test, jnp.zeros((data.ps_test.shape[0], 1)).T], axis=0)

print('X_test_treated is now:', X_test_treated)
ate_test = mvbart.predict(X_test_treated) - mvbart.predict(X_test_control)

print(ate_test.shape) # (3000, 432, 10)
ate_test_mean = jnp.mean(ate_test, axis=0) # (432, 10)
ate_test_var = jnp.var(ate_test, axis=0)
for i, ate in enumerate(jnp.mean(ate_test_mean, axis=0)):
    col_name = outcomes_names[i] if i < len(outcomes_names) else f'col_{i}'
    # print(f'ATE for {col_name}: {ate:.3f}')
    print(f'ATE for {col_name}:', ate)

# RMSE
# rmse = jnp.sqrt(jnp.mean(jnp.square(yhat_test_mean - data.y_test)))
# rmse = jnp.sqrt(jnp.mean(jnp.square(yhat_test_mean - data.y_test)))
rmse_per_col = jnp.sqrt(jnp.mean(jnp.square(yhat_test_mean - data.y_test), axis=0))
for i, rmse in enumerate(rmse_per_col):
    col_name = outcomes_names[i] if i < len(outcomes_names) else f'col_{i}'
    print(f'RMSE for {col_name}: {rmse:.3f}')
print('yhat_test is of shape', yhat_test.shape)
# print('yhat_test ptrdicted is', yhat_test_mean[1:5,])
# print('yhat_test truth is', data.y_test[1:5,])
print(f'Total RMSE: {rmse:#.2g}')

# Covariance matrix


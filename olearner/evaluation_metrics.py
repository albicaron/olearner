# Useful evaluation Metrics for simulated studies

import numpy as np
from scipy import stats as sts


# Bias
def bias(T_true, T_est):
    return np.mean(T_true.reshape((-1, 1)) - T_est.reshape((-1, 1)))


# PEHE (square root PEHE)
def PEHE(T_true, T_est):
    return np.sqrt(np.mean((T_true.reshape((-1, 1)) - T_est.reshape((-1, 1))) ** 2))


# Accuracy for PS estimation
def accur(z_true, z_hat, threshold=0.5):
    return np.mean((z_true.reshape((-1, 1)) - z_hat.reshape((-1, 1))) ** 2)


# Monte Carlo Standard Error
def MC_se(x, B):
    return sts.t.ppf(0.975, B - 1) * np.std(np.array(x)) / np.sqrt(B)


# Monte Carlo Standard Error with "NaN"
def MC_se_NaN(x, B):
    return sts.t.ppf(0.975, B - 1) * np.nanstd(np.array(x)) / np.sqrt(B)

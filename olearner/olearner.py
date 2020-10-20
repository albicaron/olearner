# Copyright (c) 2020, Alberto Caron
# Licensed under the BSD 3-clause license

import pandas as pd
import numpy as np
from scipy import stats as sts
from copulae import GaussianCopula

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# A Data Generating Process
def example_data(N=1000, P=10, rng=1):
    # Set seed
    np.random.seed(rng+100)

    # Generate X with a little bit of correlation b/w the continuous variables (i.e. 0.3 Pearson coeffincient c.a.)
    X = np.zeros((N, P))

    # Generate Gaussian copula correlated uniforms
    g_cop = GaussianCopula(dim=P)

    mycova = np.ones((P, P))

    for i in range(P):
        for j in range(P):

            mycova[i, j] = 0.3**(np.abs(i - j)) + np.where(i == j, 0, 0.1)

    g_cop[:] = mycova

    rv = g_cop.random(N)

    # Generate correlated BALANCED covariates (2 out of 10)
    X[:, 0:2] = np.asarray([sts.norm.ppf(rv[:, i]) for i in range(2)]).T

    # Generate Z
    und_lin = -0.2 + 0.5*X[:, 0] + 0.7*X[:, 1] + sts.uniform.rvs(size=N)/10
    pscore = 0.9*(sts.norm.cdf(und_lin))
    # import seaborn as sns
    # sns.distplot(pscore, bins=50)
    # from scipy import stats
    # stats.describe(pscore)

    Z = sts.binom.rvs(1, pscore)
    # np.bincount(Z)

    # Generate UNBALANCED covariates (8 out of 10, Z enters mean)
    X[:, 2:5] = np.asarray([sts.norm.ppf(rv[:, i], loc=2*Z) for i in range(2, 5)]).T
    X[:, 5:P] = np.asarray([sts.binom.ppf(rv[:, i], n=1, p=0.3 + 0.3*Z) for i in range(5, P)]).T

    # Generate Y
    mu = 6 + 0.5*X[:, 2] + 0.3*(X[:, 3] - 0.5)**2 + 1*X[:, 4]*(X[:, 9] + 1)
    ITE = 1 + 0.5*X[:, 2]
    # sns.distplot(mu)
    # sns.distplot(ITE, bins=30)

    sigma = np.std(ITE)
    Y = mu + ITE*Z + sts.norm.rvs(0, sigma, N)

    return Y, X, Z, ITE


# PEHE evaluation metric
def PEHE(T_true, T_est):
    return np.sqrt(np.mean((T_true.reshape((-1, 1)) - T_est.reshape((-1, 1))) ** 2))


# Monte Carlo Standard Error
def MC_se(x, B):
    return sts.t.ppf(0.975, B - 1) * np.std(np.array(x)) / np.sqrt(B)


class OLearner:
    """
    A O-Learner implementation with sklearn models.

    """
    def __init__(self, outcome_type="Continuous", pi_model=None, y_model=None):
        """
        Class constructor. A model object.

        :pi_model: a sklearn probabilistic classification model (default is LogisticRegression)
        :y_model: a sklearn regression/classification model (default is LinearRegression)
        :outcome_type: ["Continuous", "Binary"]
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~
        # ** input arguments
        # ~~~~~~~~~~~~~~~~~~~~~~~
        self.outcome_type = outcome_type
        self.pi_model = pi_model
        self.y_model = y_model

        if pi_model is None:
            self.pi_model = LogisticRegression()

        if y_model is None:
            if self.outcome_type == "Binary":
                self.y_model = LogisticRegression()
            else:
                self.y_model = LinearRegression()

    def fit(self, X, Y, Z, W=None):
        """
        Fit Propensity and Outcome models, and returns CATE fit model.
        X has to be an N x k matrix.

        :X: Covariates N x k numpy array for the Outcome Model
        :W: Covariates for the propensity score model (default equal to X)
        :Y: Outcome N x 1 numpy array
        :Z: Treatment Assignment N x 1 numpy array
        """
        # 1) Fit Propensity Model
        if W is None:
            W = X

        self.pi_model.fit(W, Z)
        z_hat = self.pi_model.predict_proba(W)[:, 1]

        # 2) Overlap Weighting step
        Y_overlap = Y
        Y_overlap[Z == 1] = Y[Z == 1]*(1 - z_hat[Z == 1])
        Y_overlap[Z == 0] = Y[Z == 0]*z_hat[Z == 0]

        # 3) Fit Outcome Model
        self.y_model.fit(X, Y_overlap)

    def predict(self, X):
        """
        Predict CATE for set of covariates.

        :X: Covariates N x k_1 numpy array for the Outcome Model
        :W: Covariates N x k_2 numpy array for the Outcome Model
        """
        # Predict PS
        CATE_est = self.y_model.predict(X)

        return CATE_est


class DROLearner:
    """
    A doubly-robust implementation of O-Learner with sklearn models.
    """
    def __init__(self, outcome_type="Continuous", pi_model=None, y_model=None, g_model=None):
        """
        Class constructor. A model object.

        :g_model: a sklearn regression/classification model for raw outcome model g_t(X) (Default is LinearRegression)
        :pi_model: a sklearn probabilistic classification model (Default is LogisticRegression)
        :y_model: a sklearn regression/classification model for CATE (Default is LinearRegression)
        :outcome_type: ["Continuous", "Binary"]
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~
        # ** input arguments
        # ~~~~~~~~~~~~~~~~~~~~~~~
        self.outcome_type = outcome_type
        self.pi_model = pi_model
        self.g_model = g_model
        self.y_model = y_model

        if self.pi_model is None:
            self.pi_model = LogisticRegression()

        if g_model is None:
            if self.outcome_type == "Binary":
                self.g_model = LogisticRegression()
            else:
                self.g_model = LinearRegression()

        if y_model is None:
            if self.outcome_type == "Binary":
                self.y_model = LogisticRegression()
            else:
                self.y_model = LinearRegression()

    def fit(self, X, Y, Z, W=None):
        """
        Fit Propensity and Outcome models, and returns CATE fit model.
        X has to be an N x k matrix.

        :X: Covariates N x k numpy array for the Outcome Model
        :W: Covariates for the propensity score model (default equal to X)
        :Y: Outcome N x 1 numpy array
        :Z: Treatment Assignment N x 1 numpy array
        """
        # 1) Fit raw outcome model (T-Learner)
        y1_model = self.g_model.fit(X[Z == 1, :], Y[Z == 1])
        y0_model = self.g_model.fit(X[Z == 0, :], Y[Z == 0])

        y1_fit = y1_model.predict(X[Z == 1, :])
        y0_fit = y0_model.predict(X[Z == 0, :])

        g_fit = Y
        g_fit[Z == 1] = y1_fit
        g_fit[Z == 0] = y0_fit

        # 2) Fit Propensity Model
        if W is None:
            W = X

        self.pi_model.fit(W, Z)
        z_hat = self.pi_model.predict_proba(W)[:, 1]

        # 2) Construct DR Overlap-Weighted outcome
        Y_DR = Y
        Y_DR[Z == 1] = g_fit[Z == 1] + (Y[Z == 1] - g_fit[Z == 1])*(1 - z_hat[Z == 1])
        Y_DR[Z == 0] = g_fit[Z == 0] + (Y[Z == 0] - g_fit[Z == 0])*z_hat[Z == 0]

        # 3) fit CATE
        self.y_model.fit(X, Y_DR)

    def predict(self, X):
        """
        Predict CATE for set of covariates.

        :X: Covariates N x k_1 numpy array for the Outcome Model
        :W: Covariates N x k_2 numpy array for the Outcome Model
        """
        # Predict PS
        CATE_est = self.y_model.predict(X)

        return CATE_est

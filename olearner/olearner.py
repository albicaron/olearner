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
    und_lin = -0.2 + 0.5*X[:, 0] + 0.3*X[:, 1]**2 + sts.uniform.rvs(size=N)/10
    pscore = 0.8*sts.norm.cdf(und_lin)

    Z = sts.binom.rvs(1, pscore)

    # Generate UNBALANCED covariates (8 out of 10, Z enters mean)
    X[:, 2:5] = np.asarray([sts.norm.ppf(rv[:, i], loc=Z) for i in range(2, 5)]).T
    X[:, 5:P] = np.asarray([sts.binom.ppf(rv[:, i], n=1, p=0.3 + 0.1*Z) for i in range(5, P)]).T

    # Generate Y
    mu = 6 + 0.3*X[:, 2]**2 + 1*X[:, 3] + 0.5*X[:, 4]*X[:, 8]
    ITE = 2 + 0.1*X[:, 2]**2 + 0.8*X[:, 8]

    sigma = np.std(ITE)/2
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
    Doubly-robust implementation of O-Learner with sklearn models.
    Baseline outcome model g() is learnt as in a S-Learner
    """
    def __init__(self, outcome_type="Continuous", pi_model=None, y_model=None, g_model=None):
        """
        Class constructor. A model object.

        :g_model: a sklearn regression/classification model for raw outcome model g_t(X) (Default is LinearRegression)
        :pi_model: a sklearn probabilistic classification model (Default is LogisticRegression)
        :y_model: a sklearn regression/classification model for CATE (Default is LinearRegression)
        :outcome_type: ["Continuous", "Binary"]
        """
        if outcome_type not in ["Continuous", "Binary"]:
            raise ValueError('Currently supported types are ["Continuous", "Binary"]')

        # ~~~~~~~~~~~~~~~~~~~~~~~
        # ** input arguments
        # ~~~~~~~~~~~~~~~~~~~~~~~
        self.outcome_type = outcome_type
        self.pi_model = pi_model
        self.g_model = g_model
        self.y_model = y_model

        if self.pi_model is None:
            self.pi_model = LogisticRegression(solver="lbfgs")

        if g_model is None:
            if self.outcome_type == "Binary":
                self.g_model = LogisticRegression(solver="lbfgs")
            else:
                self.g_model = LinearRegression()

        if y_model is None:
            if self.outcome_type == "Binary":
                self.y_model = LogisticRegression(solver="lbfgs")
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
        # Check inputs
        if len(X.shape) is not 2:
            raise ValueError('X must be a 2-dimensional N x k numpy array')

        if len(Y) != len(Z) != X.shape[0]:
            raise ValueError('Y, Z and X must have the same number of rows N')

        # 1) Fit raw outcome model (T-Learner)
        Y_raw_model = self.g_model.fit(np.c_[X, Z], Y)

        g1_fit = Y_raw_model.predict(np.c_[X, np.ones(X.shape[0])])
        g0_fit = Y_raw_model.predict(np.c_[X, np.zeros(X.shape[0])])

        # 2) Fit Propensity Model
        if W is None:
            W = X

        self.pi_model.fit(W, Z)
        z1_hat = self.pi_model.predict_proba(W)[:, 1]

        # 3) Construct DR Overlap-Weighted outcome
        Y1_DR = g1_fit
        Y1_DR[Z == 1] += (Y[Z == 1] - g1_fit[Z == 1])*(1 - z1_hat[Z == 1])

        Y0_DR = g0_fit
        Y0_DR[Z == 0] += (Y[Z == 0] - g0_fit[Z == 0])*z1_hat[Z == 0]

        # 4) fit CATE
        self.y_model.fit(X, Y1_DR - Y0_DR)

    def predict(self, X):
        """
        Predict CATE for set of covariates.

        :X: Covariates N x k_1 numpy array for the Outcome Model
        :W: Covariates N x k_2 numpy array for the Outcome Model
        """
        # Predict PS
        CATE_est = self.y_model.predict(X)

        return CATE_est



class OTLearner:
    """
    Doubly-robust O-Learner with sklearn models, where g() is learnt separately for Z=1 and Z=0 (as in a T-Learner).
    """
    def __init__(self, outcome_type="Continuous", pi_model=None, y_model=None, g1_model=None, g0_model=None):
        """
        Class constructor. A model object.

        :g_model: a sklearn regression/classification model for raw outcome model g_t(X) (Default is LinearRegression)
        :pi_model: a sklearn probabilistic classification model (Default is LogisticRegression)
        :y_model: a sklearn regression/classification model for CATE (Default is LinearRegression)
        :outcome_type: ["Continuous", "Binary"]
        """
        if outcome_type not in ["Continuous", "Binary"]:
            raise ValueError('Currently supported types are ["Continuous", "Binary"]')

        # ~~~~~~~~~~~~~~~~~~~~~~~
        # ** input arguments
        # ~~~~~~~~~~~~~~~~~~~~~~~
        self.outcome_type = outcome_type
        self.pi_model = pi_model
        self.g1_model = g1_model
        self.g0_model = g0_model
        self.y_model = y_model

        if self.pi_model is None:
            self.pi_model = LogisticRegression(solver="lbfgs")

        if g1_model is None:
            if self.outcome_type == "Binary":
                self.g1_model = LogisticRegression(solver="lbfgs")
            else:
                self.g1_model = LinearRegression()

        if g0_model is None:
            if self.outcome_type == "Binary":
                self.g0_model = LogisticRegression(solver="lbfgs")
            else:
                self.g0_model = LinearRegression()

        if y_model is None:
            if self.outcome_type == "Binary":
                self.y_model = LogisticRegression(solver="lbfgs")
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
        # Check inputs
        if len(X.shape) is not 2:
            raise ValueError('X must be a 2-dimensional N x k numpy array')

        if len(Y) != len(Z) != X.shape[0]:
            raise ValueError('Y, Z and X must have the same number of rows N')

        # 1) Fit raw outcome model (T-Learner)
        Y1_raw_model = self.g1_model.fit(X[Z == 1], Y[Z == 1])
        Y0_raw_model = self.g0_model.fit(X[Z == 0], Y[Z == 0])

        g1_fit = Y1_raw_model.predict(X)
        g0_fit = Y0_raw_model.predict(X)

        # 2) Fit Propensity Model
        if W is None:
            W = X

        self.pi_model.fit(W, Z)
        z1_hat = self.pi_model.predict_proba(W)[:, 1]

        # 3) Construct DR Overlap-Weighted outcome
        Y1_DR = g1_fit
        Y1_DR[Z == 1] += (Y[Z == 1] - g1_fit[Z == 1])*(1 - z1_hat[Z == 1])

        Y0_DR = g0_fit
        Y0_DR[Z == 0] += (Y[Z == 0] - g0_fit[Z == 0])*z1_hat[Z == 0]

        # 4) fit CATE
        self.y_model.fit(X, Y1_DR - Y0_DR)

    def predict(self, X):
        """
        Predict CATE for set of covariates.

        :X: Covariates N x k_1 numpy array for the Outcome Model
        :W: Covariates N x k_2 numpy array for the Outcome Model
        """
        # Predict PS
        CATE_est = self.y_model.predict(X)

        return CATE_est


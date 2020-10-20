# Script to test different types of Meta-Learners on simulated data

# Importing packages
import pandas as pd

from olearner.olearner import PEHE, example_data
from olearner.olearner import OLearner, DROLearner

from econml.metalearners import SLearner, TLearner, XLearner
from econml.drlearner import DRLearner

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier


# Options
N = 1000
myY, myX, myZ, ITE = example_data(rng=1)


# @@@@@@@@@
# Models
# @@@@@@@@@

# S-Learner
myS = SLearner(overall_model=LinearRegression())
myS.fit(Y=myY, T=myZ, X=myX)
S_fit = myS.effect(X=myX)

# T-Learner
myT = TLearner(models=LinearRegression())
myT.fit(Y=myY, T=myZ, X=myX)
T_fit = myT.effect(X=myX)

# X-Learner
myXL = XLearner(models=LinearRegression(),
                propensity_model=MLPClassifier(hidden_layer_sizes=(4, 2), max_iter=5000),
                cate_models=LinearRegression())
myXL.fit(Y=myY, T=myZ, X=myX)
XL_fit = myXL.effect(X=myX)

# DR Learner
myDR = DRLearner(model_propensity=MLPClassifier(hidden_layer_sizes=(4, 2), max_iter=5000),
                 model_regression=LinearRegression(),
                 model_final=LinearRegression())
myDR.fit(Y=myY, T=myZ, X=myX)
DR_fit = myDR.effect(X=myX)

# O-Learner
myO = OLearner(pi_model=MLPClassifier(hidden_layer_sizes=(4, 2), max_iter=5000),
               y_model=LinearRegression())
myO.fit(myX, myY, myZ)
O_fit = myO.predict(myX)


# Evaluate PEHE for each model
PEHE_final = {'S-Learner': PEHE(ITE, S_fit), 'T-Learner': PEHE(ITE, T_fit),
              'X-Learner': PEHE(ITE, XL_fit), 'DR-Learner': PEHE(ITE, DR_fit),
              'O-Learner': PEHE(ITE, O_fit)}

PEHE_final = pd.DataFrame(PEHE_final, index=["PEHE"])

print(PEHE_final)

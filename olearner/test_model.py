# Script to test different types of Meta-Learners on simulated data

# Importing packages
import pandas as pd
import numpy as np

from olearner.olearner import PEHE, example_data
from olearner.olearner import OLearner

from econml.metalearners import SLearner, TLearner, XLearner
from econml.drlearner import DRLearner
from sklearn.linear_model import LinearRegression, LogisticRegression

# Options
N, B = [1000, 200]

models = {"S": np.zeros(N), "T": np.zeros(N), "X": np.zeros(N),
          "DR": np.zeros(N), "O": np.zeros(N)}
PEHES = {"S": np.zeros(B), "T": np.zeros(B), "X": np.zeros(B),
          "DR": np.zeros(B), "O": np.zeros(B)}

# @@@@@@@@@
# Models
# @@@@@@@@@

for i in range(B):

    # Create data
    myY, myX, myZ, ITE = example_data(N, rng=i*13)

    # S-Learner
    myS = SLearner(overall_model=LinearRegression())
    myS.fit(Y=myY, T=myZ, X=myX)
    models["S"] = myS.effect(X=myX)

    # T-Learner
    myT = TLearner(models=LinearRegression())
    myT.fit(Y=myY, T=myZ, X=myX)
    models["T"] = myT.effect(X=myX)

    # X-Learner
    myXL = XLearner(models=LinearRegression(),
                    propensity_model=LogisticRegression(solver="lbfgs"),
                    cate_models=LinearRegression())
    myXL.fit(Y=myY, T=myZ, X=myX)
    models["X"] = myXL.effect(X=myX)

    # DR Learner
    myDR = DRLearner(model_propensity=LogisticRegression(solver="lbfgs"),
                     model_regression=LinearRegression(),
                     model_final=LinearRegression())
    myDR.fit(Y=myY, T=myZ, X=myX)
    models["DR"] = myDR.effect(X=myX)

    # O-Learner
    myO = OLearner()
    myO.fit(myX, myY, myZ)
    models["O"] = myO.predict(myX)

    # PEHES fill
    for j in models.keys():
        PEHES[j][i] = PEHE(ITE, models[j])


# Evaluate PEHE for each model
PEHE_final = {'S-Learner': np.mean(PEHES["S"]), 'T-Learner': np.mean(PEHES["T"]),
              'X-Learner': np.mean(PEHES["X"]), 'DR-Learner': np.mean(PEHES["DR"]),
              'O-Learner': np.mean(PEHES["O"])}

PEHE_final = pd.DataFrame(PEHE_final, index=["PEHE"])

print(PEHE_final)

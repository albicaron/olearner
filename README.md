# `olearner`: Overlap Learner for Heterogeneous Treatment Effects estimation and Policy Learning
The Overlap Learner (O-Learner) is a type of Meta-Learner 
[(Kunzel et. al., 2017](https://arxiv.org/pdf/1706.03461.pdf);
[Caron et. al., 2020)](https://arxiv.org/pdf/2009.06472.pdf) based on the work on balancing weights 
by [Li et. al. (2014)](https://arxiv.org/pdf/1404.1785.pdf), designed to estimate Individual Treatment Effects (ITE) and
carry out policy learning/optimization in a reinforcement learning contextual bandits framework. The way the O-Learner works 
compared to other "Direct Methods"  Meta-Learners (S-Learner, T-Learner, X-Learner, etc.) is that it constructs a (asymptotically)
doubly-robust ([Robins et. al., 2005](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1541-0420.2005.00377.x), [Dudik et. al., 2011](https://icml.cc/2011/papers/554_icmlpaper.pdf)) estimation procedure that makes use
of balancing properties of the "overlap weights" scheme proposed in 
[Li et. al. (2014)](https://arxiv.org/pdf/1404.1785.pdf) (instead of using Inverse Propensity Score 
Weighting - IPW - such as in the doubly-robust learner), by first estimating a model for Propensity Scores estimation. The "overlap weights" 
scheme has nice asymptotic properties and is specifically helpful to tackle unbalanced covariates 
in the treatment groups, typically found in observational studies, beside offering a more numerically stable solution than
IPW (issues arising when they are close to zero).

## Python implementation
The Python light-weight package can be found in the subfolder `olearner\olearner.py`. The 
implementation supports all the compatible Regression and Classification models of the `sklearn`
library, that can be used as "base-learners" for the O-Learner algorithm.  

The subfolder also contains the `olearner\test_model.py` file, which contains a (extremely) short 
comparison with other popular Meta-Learners on a simulated example with non-overlapping regions of the
covariates space between the two treated groups. 

The file `olearner_illustration.ipynb` briefly explains how the theory and how model is constructed.
It also guides through the simulated example of `olearner\test_model.py`, to compare O-Learner with
other methods.

## References
- Caron A., Manolopoulou I. & Baio G. (2020). *Estimating Individual Treatment Effects using Non-Parametric Regression Models: A Review*. [(pre-print)](https://arxiv.org/abs/2009.06472)

- Li F., Morgan K. L. & Zaslavsky A. M. (2018). *Balancing Covariates via Propensity Score Weighting*, Journal of the American Statistical Association, 113:521, 390-400.

- Künzel S. R., Sekhon J. S., Bickel P. J. & Yu B. (2019). *Metalearners for estimating heterogeneous treatment effects using machine learning*. PNAS March 5, 2019 116 (10) 4156-4165.

- Dudik M., Langford J. & Li L. (2011). *Doubly Robust Policy Evaluation and Learning*. [(pre-print)](https://arxiv.org/abs/1103.4601)

- Jiang N., Li L. (2016). *Doubly Robust Off-policy Value Evaluation for Reinforcement Learning*. Proceedings of The 33rd International Conference on Machine Learning, PMLR 48:652-661.

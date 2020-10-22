# `olearner`: Overlap Learner for Heterogeneous Treatment Effects estimation
The Overlap Learner (O-Learner) is a type of Meta-Learner 
[(Kunzel et. al., 2017](https://arxiv.org/pdf/1706.03461.pdf);
[Caron et. al., 2020)](https://arxiv.org/pdf/2009.06472.pdf) based on the work on balancing weights 
by [Li et. al. (2014)](https://arxiv.org/pdf/1404.1785.pdf). The way the O-Learner works 
compared to other Meta-Learners (S-Learner, T-Learner, X-Learner, etc.) is that it constructs a
balanced version of the outcomes **Y** via the "overlap weights" scheme proposed in 
[Li et. al. (2014)](https://arxiv.org/pdf/1404.1785.pdf) (instead of using Inverse Propensity Score 
Weighting - IPW), by first estimating a model for Propensity Scores estimation. The "overlap weights" 
scheme has nice asymptotic properties and is specifically helpful to tackle unbalanced covariates 
in the treatment groups, typically found in observational studies.

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
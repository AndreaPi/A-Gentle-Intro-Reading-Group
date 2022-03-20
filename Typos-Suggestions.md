##### 2.1 Classification with Adaptive Predictive Sets
**Suggestion**: in $ k=\inf\left\{ k:\sum_{i=1}^k\hat{f}(X_{test})_{\pi_i}\geq 1-\alpha \right\} $, using the same symbol $k$ both for the infimum of the sum upper bounds, and for the sum upper bounds, can be a bit confusing, so maybe consider using a different symbol?

#### 2.2 Conformalized Quantile Regression
**Suggestion**: since regression is usually done on tabular data, and boosting regressors tend to do better than NNs on tabular data you may want to mention that  scikit-learn offers the possibility to train a `GradientBoostingRegressor` [with pinball loss](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html). XGBoost usually gives better point estimates than `GradientBoostingRegressor`, but that library doesn't currently offer the possibility to train with pinball loss. However, they're working on it (see [issue #7435](https://github.com/dmlc/xgboost/issues/7435)).

##### 2.3 Conformalizing Scalar Uncertainty Estimates
**Typo**: the caption of Figure 8 is wrong (part of it is copy-pasted from Figure 6)
**Suggestion**: you may want to mention that Henrik Bostrom wrote a `sklearn`-like library [crepes](https://github.com/henrikbostrom/crepes) which offers the possibility to conformalize generic regressors using uncertainty scalars. I wasn't paid by Henrik to tell you this (I don't even know him 😀)


##### 3.2 Checking for correct coverage
###### The standard deviation of $\bar{C}$
**Typo**: "We now ~~we will~~ examine the distribution of $C_j$".
**Typo**: "Unfortunately, the distribution of $\bar{C}-$the mean of R independent beta-binomial ~~distributions~~ random variables$-$does not have a closed form".
**Typo**: "If the simulated average empirical coverage does not align well with the coverage observed **on** the real data, there is likely a problem in the conformal implementation.".

##### 3.2 Evaluating adaptiveness
##### Feature-stratified coverage metric.
**Typo**: "In words, this is the observed coverage for all units for which ~~to~~ the discrete feature takes value _g_"
**Not sure if typo**: "For example, in classification we might divide the observations _into units into three groups_". The last part is a bit unclear to me, not sure if it's a typo or not.
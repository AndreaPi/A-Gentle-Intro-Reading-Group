##### 2.1 Classification with Adaptive Predictive Sets
**Suggestion**: in $ k=\inf\left\{ k:\sum_{i=1}^k\hat{f}(X_{test})_{\pi_i}\geq 1-\alpha \right\} $, using the same symbol $k$ both for the infimum of the sum upper bounds, and for the sum upper bounds, can be a bit confusing, so maybe consider using a different symbol?

#### 2.2 Conformalized Quantile Regression
**Suggestion**: since regression is usually done on tabular data, and boosting regressors tend to do better than NNs on tabular data you may want to mention that  scikit-learn offers the possibility to train a `GradientBoostingRegressor` [with pinball loss](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html). XGBoost usually gives better point estimates than `GradientBoostingRegressor`, but that library doesn't currently offer the possibility to train with pinball loss. However, they're working on it (see [issue #7435](https://github.com/dmlc/xgboost/issues/7435)).

##### 2.3 Conformalizing Scalar Uncertainty Estimates
**Typo**: the caption of Figure 8 is wrong (part of it is copied & pasted from Figure 6)
**Suggestion**: you may want to mention that Henrik Bostrom wrote a `sklearn`-like library [crepes](https://github.com/henrikbostrom/crepes) which offers the possibility to conformalize generic regressors using uncertainty scalars. I wasn't paid by Henrik to tell you this (I don't even know him 😀)

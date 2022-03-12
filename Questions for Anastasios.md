##### 1 Conformal Prediction, Remarks
**Q**: you note that $\mathcal{T}(x)$ is _adaptive_ to the test input, since it's a function of $x$, and it gets bigger when the input is hard and/or the model is uncertain. How can we see that in the case of the softmax classifier? Does the following make sense? $s(x,y)=1-\hat{f}(x)_{y}$:  if the model is uncertain, the softmax values will tend to be the same for all labels ($\hat{f}(X_{test})_{y}\to\frac 1 K \ \forall y$) and the conformal scores will likely be all $1-\frac 1K \geq\hat{q}\to\mathcal{T}(X_{test})=\{1,\dots,K\}$. This is not a rigorous argument (we're not sure that $1-\frac 1K \geq\hat{q}$), but maybe it could be made rigorous?
**Q**: You note that $\mathcal{T}(x)$ can be interpreted as a set of plausible classes for $X_{test}$. Can we say, in statistical parlance that $\mathcal{T}(x)$ is the set of all labels for which we fail to reject the null hypothesis $H_0$:"$(X_{test},y)$ is a correct pairing"? The conformal scores $s(X_{test},y)$ would then be the $p-$values under $H_0$ of each label $y$.

##### 2.1 Classification with Adaptive Predictive Sets
**Q**:"[..]_if the softmax outputs $\hat{f}(X_{test})$ were a perfect model of $Y_{test}|X_{test}$_ [..]": wouldn't it be more appropriate to consider them a perfect model of $\mathbb{P}(Y_{test}|X_{test})$, the conditional probability of $Y_{test}$ given $X_{test}$?
**Suggestion**: in $ k=\inf\left\{ k:\sum_{i=1}^k\hat{f}(X_{test})_{\pi_i}\geq 1-\alpha \right\} $, using the same symbol $k$ both for the infimum of the sum upper bounds, and for the sum upper bounds, can be a bit confusing, so maybe consider using a different symbol?
**Q**: concerning the expression
$$\begin{equation}
\mathcal{T}(X_{test})=\{\pi_1,\dots,\pi_k \} \ \text{where} \ k=\inf\left\{ k:\sum_{i=1}^k\hat{f}(X_{test})_{\pi_i}\geq \hat{q} \right\} 
\end{equation}$$

for the prediction set, I was a bit surprised because usually we include all labels $\pi_i$ such that $s(X_{test},\pi_i)\leq\hat{q}$, instead of $\geq\hat{q}$. Is the following reasoning correct? **For all labels $\pi_i$ with $i<k$,** $s(X_{test},\pi_i)$ must be $\leq\hat{q}$, because according to (1) $k$ is the smallest integer such that $s(X_{test},\pi_k)\geq\hat{q}$. I guess we include also $\pi_k$ to handle the corner case when $s(X_{test},\pi_k)$ is exactly equal to $\hat{q}$

#### 2.2 Conformalized Quantile Regression
**Suggestion**: since regression is usually done on tabular data, and boosting regressors tend to do better than NNs on tabular data you may want to mention that  scikit-learn offers the possibility to train a `GradientBoostingRegressor` [with pinball loss](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html). XGBoost usually gives better point estimates than `GradientBoostingRegressor`, but that library doesn't currently offer the possibility to train with pinball loss. However, they're working on it (see [issue #7435](https://github.com/dmlc/xgboost/issues/7435)).

##### 2.3 Conformalizing Scalar Uncertainty Estimates
**Q**: if I understand correctly, Conformalized regression can be applied to any model which either outputs an uncertainty scalar (together with the point prediction), or can be made to do so with minor modifications. This means that it could be applied to correct the well-known deficiencies of MC Dropout. Is this correct? Well, at least it would fix the coverage deficiency - it would still be worse that using Conformalized Quantile Regression.
**Typo**: the caption of Figure 8 is wrong (part of it is copied & pasted from Figure 6)
**Suggestion**: you may want to mention that Henrik Bostrom wrote a `sklearn`-like library [crepes](https://github.com/henrikbostrom/crepes) which offers the possibility to conformalize generic regressors using uncertainty scalars. I wasn't paid by Henrik to tell you this (I don't even know him 😀)

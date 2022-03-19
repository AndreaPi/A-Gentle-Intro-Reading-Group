# A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification

http://arxiv.org/abs/2107.07511

### 1 Conformal Prediction

Given a black box Machine Learning model $\hat{f}: X\mapsto Y$, and a confidence level $\alpha\in[0,1]$, we want to construct a **_valid_** _$\alpha-$confidence set_ $\mathcal{T}(X)$, i.e., a set of labels that, for each test sample $X_{test}$, cointains the true label with probability at least $1-\alpha$:

$$ \begin{equation}
 1-\alpha \leq \mathbb{P}(Y_{test} \in\mathcal{T}(X_{test})) \leq 1-\alpha +\frac{1}{n+1} 
 \end{equation}$$

We want to our method to be:
- distribution-free (no assumptions on $\mathbb{P}(X,Y)$
- model-agnostic (valid for linear models, RFs, CNNs, etc.)

We assume that we have access to _exchangeable_ samples $(X_i,Y_i)_{i=1,...,n}$ from the unknown $\mathbb{P}(X,Y)$ (i.i.d. data are exchangeable), not used for the training of $\hat{f}$ (a _calibration set_ $S$). We also assume that $\hat{f}$ encodes an heuristic notion of uncertainty (i.e., it's a probabilistic model). This is the case with a ResNet model for $K-$class classification, for example, whose output is a softmax vector $\hat{f}(X)\in[0,1]^K,\ \sum_{i=1}^K\hat{f}_i(X)=1$. These softmax scores can be interpreted as the "confidence" of the model in attributing class $i$ to $X$: of course we know that this notion is heuristic, and that a label predicted with 90% confidence won't actually be correct on average 90% of the times. We thus need to correct them using the CP algorithm.

To build $\mathcal{T}(X)$ from $\hat{f}, S$, we use the following process:
- we define a _conformal score function_, i.e a function $s(x,y,\hat{f})\mapsto\mathbb{R}$ which should encode disagreement between the true label of $X$, and the label predicted by $\hat{f}$. A larger conformal score means that the predicted label _does not conform_ to the true label, which is a bit counterintuitive. This is why https://arxiv.org/abs/2202.07650 call it a _nonconformity measure_ (the higher the measure, the less the test point conforms). In the case of image classification, we can define $s(x,y,\hat{f})=1-\hat{f}(x)_{y}$, i.e., the softmax score corresponding to label $y$
- we compute the conformal scores of the calibration points, $s_i=1-\hat{f}(X_i)_{Y_i},\ i=1,\dots n$, and the $\frac{\left\lceil(n+1)(1-\alpha)\right\rceil}{n}-$quantile $\hat{q}$ of the conformal scores $s_i$ (similar to the $(1-\alpha)-$quantile with a small correction). Intuitively, $1-\alpha$ of the true labels have a conformal score which is below $\hat{q}$ (this can be proven rigorously).
- now, for a new test point $X_{test}$, we compute the conformal score $s(X_{test},y,\hat{f})$ for each possible label, and we define 

$$\mathcal{T}(X_{test})=\{y:s(X_{test},y,\hat{f})\leq\hat{q}\}$$

Since $\hat{q}$ is the $\frac{\left\lceil(n+1)(1-\alpha)\right\rceil}{n}-$quantile of the correct pairs $(X_i, Y_i)$, the true label $Y_{test}$ is less than $\hat{q}$ with probability $1-\alpha$. Thus we proved that

$$ \mathbb{P}(Y_{test} \in\mathcal{T}(X_{test})) \geq 1-\alpha$$

**Remarks**
- $\mathcal{T}(x)$ is _adaptive_ to the test input, since it's a function of $x$. It gets bigger when the input is hard and/or the model is uncertain. For example, in the case  $s(x,y,\hat{f})=1-\hat{f}(x)_{y}$,  if the model is uncertain, the softmax values will tend to be the same for all labels ($\hat{f}(X_{test})_{y}\to\frac 1 K \ \forall y$) and the conformal scores will likely be all $1-\frac 1K \geq\hat{q}\to\mathcal{T}(X_{test})=\{1,\dots,K\}$
- $\mathcal{T}(x)$ can be interpreted as a set of plausible classes for $X_{test}$. **To be more precise**, we can say that $\mathcal{T}(x)$ is the set of all labels for which we fail to reject the null hypothesis $H_0$:"$(X_{test},y)$ is a correct pairing". The conformal scores are the $p-$values under $H_0$ of each label $y$
- $\mathcal{T}(x)$ is _valid_ $\triangleq$ it has guaranteed coverage $1-\alpha$ even for finite sample sizes. In other words, we have
    $$ \mathbb{P}(Y_{test} \in\mathcal{T}(X_{test})) \geq 1-\alpha$$
even if the calibration set is finite.

#### 1.1 Instructions for Conformal Prediction
It's basically what we already said, but it adds that equation (1) has a formal proof if $(X_i,Y_i)_{i=1,...,n}$ are iid samples or even just exchangeable. 
**Choice of score function**
$\mathcal{T}(x)$ built with conformal prediction, is valid for any choice of score function, even if for example $\hat{f}$ is the random classifier! However, certain choices may lead to $\mathcal{T}(x)$ which don't depend on $x$: for example they're always equal to the set of all labels, and thus useless. Although the coverage guarantee is always valid, **the usefulness of the prediction sets is determined mainly by the score function**.

### 2 Examples of Conformal Prediction 
#### 2.1 Classification with Adaptive Predictive Sets
The algorithm shown before generates prediction sets of smallest average size, but they tend to over-cover simple examples and undercover hard examples. _Adaptive prediction sets_ solve this issue. 

If the softmax scores $\hat{f}(x)_y$were perfect models of $\mathbb{P}(Y_{test}=y|X_{test}=x)$, then we could build a valid prediction set just by ordering the labels by softmax scores, and including the top-scoring ones unless the total probability mass is $\geq 1-\alpha$:

$$\{\pi_1,\dots,\pi_k \}\ \text{where} \ k=\inf\left\{ k:\sum_{i=1}^k\hat{f}(X_{test})_{\pi_i}\geq 1-\alpha \right\} $$

They're not perfect models, though, so we use conformal prediction to turns this heuristic notion of uncertainty in a rigorous one. Let

$$s(x,y) = \sum_{i=1}^k\hat{f}(X_{test})_{\pi_i} \ \text{where} \ y=\pi_k$$

Then, according to the conformal prediction algorithm to form prediction sets, we compute $\hat{q}=\text{Quantile}(s_1,\dots,s_n; \frac{\left\lceil(n+1)(1-\alpha)\right\rceil}{n})$, and we define $\mathcal{T}(X_{test})=\{y: s(X_{test},y)\leq\hat{q} \}$. This results in the following prediction set:

$$\begin{equation}
\mathcal{T}(X_{test})=\{\pi_1,\dots,\pi_k \} \ \text{where} \ k=\inf\left\{ k:\sum_{i=1}^k\hat{f}(X_{test})_{\pi_i}\geq \hat{q} \right\} 
\end{equation}$$

In fact, **for all labels $\pi_i$ with $i<k$,** $s(X_{test},\pi_i)$ must be $\leq\hat{q}$, because according to (2) $k$ is the smallest integer such that $s(X_{test},\pi_k)\geq\hat{q}$.

#### 2.2 Conformalized Quantile Regression
- fit two quantile regression models, $\hat{t}_{\alpha/2}(x)$ and $\hat{t}_{1-\alpha/2}(x)$. If they were perfect, then $[\hat{t}_{\alpha/2}(x)$,$\hat{t}_{1-\alpha/2}(x)]$ would have exact coverage $1-\alpha$
- we calibrate it using conformal prediction
- $s(x,y) = \max\{\hat{t}_{\alpha/2}(x)-y,y-\hat{t}_{1-\alpha/2}(x)\}$
- $\hat{q}=\text{Quantile}(s_1,\dots,s_n; \frac{\left\lceil(n+1)(1-\alpha)\right\rceil}{n})$
- then the prediction set is
$$\begin{equation}
\mathcal{T}(X_{test})=\{y:s(X_{test},y)\leq\hat{q}\}=[\hat{t}_{\alpha/2}(x)-\hat{q},\hat{t}_{1-\alpha/2}(x)+\hat{q}]\end{equation}$$

Proof of  second equality: $s(X_{test},y) = \max\{\hat{t}_{\alpha/2}(X_{test})-y,y-\hat{t}_{1-\alpha/2}(X_{test})\}=\max\{c_1-y,y-c_2\}$, where $c_1$ and $c_2$ are two constants for a fixed $X_{test}$.
 1. if $\max\{c_1-y,y-c_2\}=c_1-y$, then $c_1-y\leq\hat{q}\implies c_1-\hat{q}\leq y$
 2. if $\max\{c_1-y,y-c_2\}=y-c_2$, then $y-c_2\leq\hat{q}\implies y\leq c_2+\hat{q}$
Q.E.D.

Thus we expand or shrink the heuristic prediction set $[\hat{t}_{\alpha/2}(x)$,$\hat{t}_{1-\alpha/2}(x)]$ by $\hat{q}$ to achieve coverage. Note that since the correction $\hat{q}$ to the heuristic prediction set is independent of $x$, we may undercover for some $x$ and over-cover for some other $x$, but the _marginal coverage_ (on average over all $x$) will still be valid.

**Advantages of quantile regression**
- the QR intervals have good coverage even before applying CP
- they also have asymptotically valid conditional coverage

**Disadvantages of quantile regression**
- the black box model must be retrained using _pinball (quantile) loss_, so we can't just wrap a pre-trained model with CP, as we did before. Also, while for some models (NN) it's easy to retrain with a different loss, other models (such as XGBoost) don't currently offer the possibility to train with pinball loss (though see [issue #7435](https://github.com/dmlc/xgboost/issues/7435)). Since regression is often done on tabular data, and boosting methods tend to do better than NNs on tabular data, this is a limit. However scikit-learn offers the possibility to train a `GradientBoostingRegressor` [with pinball loss](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html).

#### 2.3 Conformalizing Scalar Uncertainty Estimates
Suppose we have a model that outputs a point prediction $\hat{f}(x)$ as well a scalar estimate of uncertainty $u(x)$. Examples:

- a Gaussian Process, which predicts a posterior mean $\mu(x)$ and a posterior standard deviation $\sigma(x)$
- a NN trained with `GaussianNLLLoss`

In both cases we choose to model $Y_{test}|X_{test}=x\sim\mathcal{N}(\mu(x),\sigma(x))$. This assumption is in general not valid, and thus the notion of uncertainty will be only heuristic. As always, we can correct it with CP:

- $s(x,y) =\frac{|y-\hat{f}(x)|}{u(x)}$. Note that the bigger the disagreement between $\hat{f}(x)$ and $y$, the bigger $s(x,y)$, thus this is a good score function.
- $\hat{q}=\text{Quantile}(s_1,\dots,s_n; \frac{\left\lceil(n+1)(1-\alpha)\right\rceil}{n})$
- $$\begin{equation}
\mathcal{T}(X_{test})=\{y:s(X_{test},y)\leq\hat{q}\}=[\hat{f}(x)-u(x)\hat{q},\hat{f}(x)+u(x)\hat{q}]
\end{equation}$$

Proof: same as for CQR.

**Advantages of conformalized regression based on uncertainty scalars**
- no need to retrain the model,  as long as it outputs an uncertainty scalar together with the point prediction (or it can be made to do so with minor modifications, e.g. MC Dropout)
- `sklearn`-like library available [crepes](https://github.com/henrikbostrom/crepes)
  
**Disadvantages of conformalized regession based on uncertainty scalars**
- symmetric prediction sets (though this could be fixed)
- all the adaptivity of the prediction set is given by $u(x)$, which is defined _a-priori_ by the user. In CQR, instead, the relationship between upper and lower quantile is learned from the data. As a consequence, CQR is usually more efficient (it generates prediction sets of smaller average size)
- unlike $\hat{t}_{\alpha/2}(x)$ and $\hat{t}_{1-\alpha/2}(x)$, $u(x)
$ doesn't depend on $\alpha$, so it doesn't scale correctly when we change it

#### 2.3 Conformalizing Bayes
Not too interesting, except for the **Remarks** where they note that for classification or regression, where it makes sense to talk about coverage, CP is a simple, pragmatic and computationally very efficient way to generate valid prediction sets. However, for *structured prediction*, where the output is not just an integer or a real number, but it's a complex object (instance segmentation, language modeling, protein folding, multilabel classification, etc.), the concept of coverage is ill-defined, and we must generalize CP to provide distribution-free UQ.


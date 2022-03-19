##### 1 Conformal Prediction, Remarks
**Q**: you note that $\mathcal{T}(x)$ is _adaptive_ to the test input, since it's a function of $x$, and it gets bigger when the input is hard and/or the model is uncertain. How can we see that in the case of the softmax classifier? Does the following make sense? $s(x,y)=1-\hat{f}(x)_{y}$:  if the model is uncertain, the softmax values will tend to be the same for all labels ($\hat{f}(X_{test})_{y}\to\frac 1 K \ \forall y$) and the conformal scores will likely be all $1-\frac 1K \geq\hat{q}\to\mathcal{T}(X_{test})=\{1,\dots,K\}$. This is not a rigorous argument (we're not sure that $1-\frac 1K \geq\hat{q}$), but maybe it could be made rigorous?
**Q**: You note that $\mathcal{T}(x)$ can be interpreted as a set of plausible classes for $X_{test}$. Can we say, in statistical parlance that $\mathcal{T}(x)$ is the set of all labels for which we fail to reject the null hypothesis $H_0$:"$(X_{test},y)$ is a correct pairing"? The conformal scores $s(X_{test},y)$ would then be the $p-$values under $H_0$ of each label $y$.

##### 2.1 Classification with Adaptive Predictive Sets
**Q**:"[..]_if the softmax outputs $\hat{f}(X_{test})$ were a perfect model of $Y_{test}|X_{test}$_ [..]": wouldn't it be more appropriate to consider them a perfect model of $\mathbb{P}(Y_{test}|X_{test})$, the conditional probability of $Y_{test}$ given $X_{test}$?
**Q**: concerning the expression
$$\begin{equation}
\mathcal{T}(X_{test})=\{\pi_1,\dots,\pi_k \} \ \text{where} \ k=\inf\left\{ k:\sum_{i=1}^k\hat{f}(X_{test})_{\pi_i}\geq \hat{q} \right\} 
\end{equation}$$

for the prediction set, I was a bit surprised because usually we include all labels $\pi_i$ such that $s(X_{test},\pi_i)\leq\hat{q}$, instead of $\geq\hat{q}$. Is the following reasoning correct? 
**For all labels $\pi_i$ with $i<k$,** $s(X_{test},\pi_i)$ must be $\leq\hat{q}$, because according to (1) $k$ is the smallest integer such that $s(X_{test},\pi_k)\geq\hat{q}$. I guess we include also $\pi_k$ to handle the corner case when $s(X_{test},\pi_k)$ is exactly equal to $\hat{q}$

#### 2.2 Conformalized Quantile Regression
TODO
**Q**: naïve question, but please indulge me 🙂 is the following derivation for the CQR prediction set correct? 
$$\begin{equation}
\begin{split}
\mathcal{T}(x)&=\{y:s(x,y) \leq \hat{q} \}= \{y:\max(\hat{t}_{\alpha/2}(x)-y,y-\hat{t}_{1-\alpha/2}(x))\leq \hat{q}\} \\ &= \{y:\max(c_1-y,y-c_2)\leq \hat{q}\}
\end{split}
\end{equation}$$
Now, 
 1. if $\max\{c_1-y,y-c_2\}=c_1-y$, then $c_1-y\leq\hat{q}\implies c_1-\hat{q}\leq y$
 2. if $\max\{c_1-y,y-c_2\}=y-c_2$, then $y-c_2\leq\hat{q}\implies y\leq c_2+\hat{q}$
   
Thus $\mathcal{T}(x)=[\hat{t}_{\alpha/2}(x)-\hat{q},\hat{t}_{1-\alpha/2}(x)+\hat{q}] \quad \square.$ Correct?

**Q**: if I understand correctly, Conformalized regression can be applied to any model which either outputs an uncertainty scalar (together with the point prediction), or can be made to do so with minor modifications. This means that it could be applied to correct the well-known deficiencies of MC Dropout. Is this correct? Well, at least it would fix the coverage deficiency - it would still be worse that using Conformalized Quantile Regression.

##### 2.3 Conformalizing Scalar Uncertainty Estimates
**Q**: if I understand correctly, Conformalized regression can be applied to any model which either outputs an uncertainty scalar (together with the point prediction), or can be made to do so with minor modifications. This means that it could be applied to correct the well-known deficiencies of MC Dropout. Is this correct? Well, at least it would fix the coverage deficiency - it would still be worse that using Conformalized Quantile Regression.

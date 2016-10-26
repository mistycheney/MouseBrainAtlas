# Bayesian Formulation of the Atlas Building Problem

<!-- $$
\begin{aligned}
P(\{\Theta_k\} | \{F_k\}) &= \prod_k \mathbf{P}(\Theta_k | F_k) \\ &\propto \prod_k \mathbf{P}(\Theta_k) \mathbf{P}(F_k|\Theta_k) \\
&\propto \prod_k \mathbf{P}(\Theta_k) \int dx_k \; da \; \mathbf{P}(F_k|a, x_k) \mathbf{P}(x_k| \Theta_k) \mathbf{P}(a) \\
&= \int da \; \mathbf{P}(a) \prod_k \int  \mathbf{P}(F_k|a, x_k) \mathbf{P}(x_k | \Theta_k) \mathbf{P}(\Theta_k) d{x_k}\\
\end{aligned}
$$ -->
![image](simplified_atlas_estimation_graphical_model.png)

 The only observables are the detector score volumes for all structures and all brains $\{F_{i,k}\}$.

The parameters to estimate are
- the global (affine) transform for each brain $\{A_i\}$, and,
- distribution parameters for every structure $\{\Theta_k\}$.

$\{X_{i,k}\}$ are the actual shape and position of the structure $k$ in brain $i$ in the standard space. They are sampled from the structure-specific distributions. The difference between $\{X_{i,k}\}$ of the same structure in different brain is considered to be biological. Therefore We call $\{X_{i,k}\}$ the phenotypes.


For simpler notation, we focus on the model for one brain and drop the index $i$.
![image](simplified_atlas_estimation_graphical_model_onebrain.png)

One possible setting for $\Theta_k$ is a rigid transform. In this case the distribution of every structure is characterized by a random translation/rotation of a fixed 3D shape about a mean position. This is our current setting - the fixed structure shapes and the mean positions are derived from the average of the two annotated brains.

To estimate $A$ and $\{\Theta_k\}$ given the detector score volumes $\{f_k\}$, we do the following two steps.

First, give an initial estimate for $\{\Theta_k\} = \{\hat{\theta_k}\}$, compute the maximum likelihood estimate for the global transform $\hat{a}$.

$$
\begin{align}
\hat{a} &= \text{argmax} \; \mathbf{P}(\{f_k\}| a, \{\hat{\theta_k}\}) \\
&=
\prod_k
\mathbf{P}
(f_k | a, \hat{\theta_k}) \\
&=
\prod_k
\int
\mathbf{P}(f_k | a, x_k)
\mathbf{P}(x_k | \hat{\theta_k})
d {x_k} \\
&\approx
\prod_k
\mathbf{P}\left(f_k | a, \int \mathbf{P}(x_k | \hat{\theta_k}) dx_k\right)
 \\
& =
\prod_k
\mathbf{P}(f_k | a, \mathbb{E}[X_k | \hat{\theta_k}])
\end{align}
$$

Denote $\mathbb{E}[X_k | \hat{\theta_k}]$ by $\hat{x_k}$. We call these _probabilistic structures_.

If we let the likelihood $\mathbf{P}(f_k | a, \hat{x_k})$ relate to the exponential of per-structure overlap score $S$,
$$
\mathbf{P}(f_k | a, \hat{x_k}) \propto \exp S(f_k, a, \hat{x_k})
$$

then
$$
\begin{aligned}
\log
\prod_k
\mathbf{P}(f_k | a, \hat{x_k})
&=
\sum_k \log \mathbf{P}(f_k | a, \hat{x_k}) \\
&=
\sum_k S(f_k, a, \hat{x_k}) +
\text{constant}\\
\end{aligned}
$$

The estimate $\hat{a}$ is thus the transform that maximizes the sum of per-structure overlap scores.


We can introduce a prior over global transforms, $\mathbf{P}(a)$,
the objective function then becomes

$$J(a) = \sum_k S(f_k, a, \hat{x_k}) + \log \mathbf{P}(a)$$

If the prior is Gaussian,
$$
\mathbf{P}(a) = \mathcal{N}(a; \mu_A, \Sigma_A)
$$
then

$$J(a) = \sum_k S(f_k, a, \hat{x_k}) + (a-\mu_A)^T\Sigma_A^{-1}(a-\mu_A)$$

Alternatively, we can base the prior on the observed Fisher information $\mathcal{I}(\hat{a})$. One example is the Jeffery's prior,
$\mathbf{P}(a) = \sqrt{\det \mathcal{I}(a)}$

The Fisher information is the expected value of the second derivative of log likelihood function:

$$
\begin{aligned}
\mathcal{I}(\hat{a})
&= -\sum_k \frac{\partial^2}{\partial a^2} \log \mathbf{P}(f_k | \hat{a}, \hat{\theta_k}) \\
&= -\frac{\partial^2}{\partial a^2} \sum_k S(f_k, \hat{a}, \hat{\theta_k}) \\
&= -H(\hat{a})
\end{aligned}
$$

This is the Hessian matrix of the scores.




In our current setting, the initial estimate $\{\hat{\theta_k}\}$ are all identity transforms. Global transform is estimated assuming structures are at their mean positions.

Then fix $\hat{a}$, update the structure distribution parameters $\{\Theta_k\}$,



$$
\begin{aligned}
\hat{\theta_k}
&= \text{argmax} \; \mathbf{P}(\theta_k | \hat{a}, f_k) \\
&\propto
\int d x_k
\mathbf{P}(f_k | \hat{a}, x_k)
\mathbf{P}(x_k | \theta_k)
\mathbf{P}(\theta_k) \\
&=
\int d x_k
\mathbf{P}(\hat{g_k} | x_k)
\mathbf{P}(x_k | \theta_k)
\mathbf{P}(\theta_k) \\
&\approx
\mathbf{P}\left(\hat{g_k} | \int \mathbf{P}(x_k | \theta_k) dx_k \right)
\mathbf{P}(\theta_k) \\
&=
\mathbf{P}(\hat{g_k} | \mathbb{E}[x_k | \theta_k])
\mathbf{P}(\theta_k) \\
\end{aligned}
$$

Here $\hat{g_k}$ is $f_k$ transformed by the inverse of $\hat{a}$.


<!-- # Quantify Uncertainty

We estimate the confidence intervals of $A$ and $\{\Theta_k\}$ by computing the observed Fisher information.

The estimate for the global affine transform $A$ is $\hat{a} \pm cI_n(\hat{a})^{-1/2}$.

$$
\begin{aligned}
I_n(\hat{a}) &= -\log \frac{\partial^2}{\partial a^2}\mathbf{P}(\{f_k\}|\hat{a}, \{\hat{\theta_k}\}) \\
&= -\sum_k \frac{\partial^2}{\partial a^2} \log \mathbf{P}(f_k | \hat{a}, \hat{\theta_k}) \\
&= -\frac{\partial^2}{\partial a^2} \sum_k S(f_k, \hat{a}, \hat{\theta_k}) \\
&= -H(\hat{a})
\end{aligned}
$$ -->

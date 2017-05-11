# Bayesian Formulation of the Atlas Building Problem


<!-- $$
\begin{aligned}
P(\{\Theta_k\} | \{F_k\}) &= \prod_k \mathbf{P}(\Theta_k | F_k) \\ &\propto \prod_k \mathbf{P}(\Theta_k) \mathbf{P}(F_k|\Theta_k) \\
&\propto \prod_k \mathbf{P}(\Theta_k) \int dx_k \; da \; \mathbf{P}(F_k|a, x_k) \mathbf{P}(x_k| \Theta_k) \mathbf{P}(a) \\
&= \int da \; \mathbf{P}(a) \prod_k \int  \mathbf{P}(F_k|a, x_k) \mathbf{P}(x_k | \Theta_k) \mathbf{P}(\Theta_k) d{x_k}\\
\end{aligned}
$$ -->

The parameters to estimate are
- global (affine) transform $A$, and,
- distribution parameters for every structure $\{\Theta_k\}$.

One possible setting for $\Theta_k$ is a rigid transform. In this case the distribution of every structure is characterized by a random translation/rotation of a fixed 3D shape about a mean position. This is our current setting - the fixed structure shapes and the mean positions are derived from the average of the two annotated brains.

To estimate $A$ and $\{\Theta_k\}$ given the detector score volumes $\{f_k\}$, we do the following two steps.

First, give an initial estimate for $\{\Theta_k\} = \{\hat{\theta_k}\}$, estimate the global transform $\hat{a}$. Denote by $x_k$ the *biological* shape and position (or phenoptype) of structure $k$.

$$
\begin{aligned}
\hat{a} &= \text{argmax} \; \mathbf{P}(a | \{\hat{\theta_k}\}, \{f_k\}) \\
&\propto
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
\end{aligned}
$$

Here $\mathbb{E}[x_k | \theta_k]$ are the _probabilistic structures_.

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


# Quantify Uncertainty

We estimate the confidence intervals of $A$ and $\{\Theta_k\}$ by computing the observed Fisher information.

The estimate for the global affine transform $A$ is $\hat{a} \pm cI_n(\hat{a})^{-1/2}$.

$$
\begin{aligned}
I_n(\hat{a}) &= -\log \frac{\partial^2}{\partial a^2}\mathbf{P}(\{f_k\}|\hat{a}, \{\hat{\theta_k}\}) \\
&= -\sum_k \frac{\partial^2}{\partial a^2} \log \mathbf{P}(f_k | \hat{a}, \hat{\theta_k}) \\
&= -\frac{\partial^2}{\partial a^2} \sum_k S(f_k, \hat{a}, \hat{\theta_k}) \\
&= -H(\hat{a})
\end{aligned}
$$

$$
\begin{aligned}



\end{aligned}
$$

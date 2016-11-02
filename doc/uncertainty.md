# Uncertainty Analysis of the Atlas Building Problem

Suppose the atlas starts with a mean centroid location and mean shape for each structure. More precisely, the atlas defines a set of canonical structures $\{\bar{V_k}\}$, where each $\bar{V_k}$ is a set of voxel locations for structure $k$. The locations are defined with respect to the atlas coordinate space $\Omega^0$.

For each brain $i$, the degrees of freedom are,
- individual structure transform $\{B_k^i\}$. It includes shape variation $\{Q^i_k\}$ and local 3D rigid transform of the centroid $\{R^i_k\}$ (6 parameters)
- global 3D affine transform $A^i$ (12 parameters)
- texture detector functions for each structure $\{f_k\}$.
Denote the image volume of brain $i$ by $\Omega^i$, these detectors give score volumes $\{F^i_k\}$. Each score volume $F^i_k(p) = f_k(I_p), \forall p\in \Omega^i$, where $I_p$ is the patch centered at location $p$ on the corresponding 2D section image.

and the scores include:
- texture score for each structure $\{S_k\}$, and the overall texture score defined as the sum of per-structure scores, $S = \sum_k S_k$.
- each structure's deviation from normal position $\{D_k\}$
- deviations from normal global transform $E$

For a given set of parameters, the per-structure texture score $S_k$ is defined as the sum of scores at voxels occupied by the transformed structure,

$$
\begin{align}
S_k(A^i, \{B^i_k\}, \{F^i_k\})
&= \sum_{p \in V_k(B^i_k)} F^i_k(q(p; A^i)) \\
&= \sum_{v \in \bar{V_k}} F^i_k(q(p(v; B^i_k); A^i)) \\
\end{align}
$$

Here $p(v; B^i_k)$ maps a voxel location $v$ in atlas space to another location, according to the transform $B_k^i$. $V_k(B^i_k)$ is the set of locations that results from transforming all voxels in the canonical set $\bar{V_k}$.

$q(p; A^i)$ is the affine transform function that maps a location $p$ in atlas space $\Omega^0$ to a location in subject space $\Omega^i$, according to parameters $A^i$.


Now we discuss the case for one brain. For notation clarity, we drop the index $i$ in the following discussion.

Gradients of the texture score are,

$$
\begin{align}
\frac{\partial}{\partial A} S_k = \sum_{p \in V_k(R_k, Q_k)} \frac{\partial F_k(q)}{\partial q} \frac{\partial q(\hat{p}; A)}{\partial A}
\end{align}
$$

Let $B = \{R_k, Q_k\}$ describe a combined individual transform that includes rigid transform, scaling and potentially other deformation.

$$
\begin{align}
\frac{\partial}{\partial B_k}S_k =
\sum_{v\in \bar{V_k}}
\frac{\partial F_k(q)} {\partial q}
\frac{\partial q(p; \hat{A})} {\partial p}
\frac{\partial p(v; B_k)}{\partial B_k}
\end{align}
$$



## Uncertainty for global transform $A$

If everything else are fixed, the overall score with respect to the global transform $A$ is,
$$J(A) = S(A) - \lambda E(A)$$
where texture score $S$ is as previously defined, and the global transform penalty is a quadratic term,
$$E(A) = (A - \mu_A)^T \Sigma_A^{-1} (A - \mu_A)$$

The parameters $\mu_A$ and $\Sigma_A$ are learned from data. The details will be described below.

The optimal estimate $a^*$ is the maximizer of $J(A)$,
which is found using gradient descent. The gradients of score volumes $\{F_k\}$ are pre-computed to speed up the optimization.

To quantify the uncertainty, we also compute the Hessian,
$$
H_A(a^*) =
\frac{\partial^2}{\partial A^2} J(a^*) =
\frac{\partial^2}{\partial A^2} S(a^*) - \lambda \Sigma_A^{-1}
$$

The Hessian of $S$ can be computed either by drawing on pre-computed second derivatives of score volumes $\{F_k\}$, or by numerical differentiation around $a^*$.

Based on $H_A$, we can construct confidence interval $[u_{\gamma,A}, v_{\gamma,A}]$ with confidence level $\gamma$, centered at $a^*$.

The confidence intervals $[u^\gamma_{A^i}, v^\gamma_{A^i}]$ for all brains are then aggregated to give an estimate for $\mu_A$ and $\Sigma_A$.

## Uncertainty for individual transforms $B_k$

If $R_k$ is rigid transform, and $Q_k$ is scaling and shearing, then the combined effect can be characterized by an 12 parameter affine transform $B_k$.
The computed Hessian will be with respect to 12 parameters.

To extract intuitive meaning from the Hessian,
say the scale of the
we can compute the  

$$\frac{\partial^2}{\partial g^2} g(B)$$





## Uncertainty for individual centroid transform $R_k$

Assume the centroids of different structures are independent, the overall score with respect to $R_k$ is,
$$J(R_k) = S(R_k) - \lambda D(R_k)$$

The deviation penalty is also a quadratic term,
$$D(R_k) = (R_k - \mu_{R_k})^T \Sigma_{R_k}^{-1} (R_k - \mu_{R_k})$$

Optimization also uses gradient descent.

The Hessian is ,
$$H_{R_k}(r_k^*) = \frac{\partial^2}{\partial A^2}S(r_k^*) - \lambda \Sigma_{R_k}^{-1} $$

## Uncertainty for individual shape transform $Q_k$

If $Q_k$ is parameterized by scaling $s$,

# Quantifying uncertainty using Z-Scores and Hessians

![computation diagram](yoav_diagram.png)

We discuss only the per-landmark transforms in this report.

Suppose the estimate for the parameter vector is $\theta^*$ and the score is $f^*$. Take a region centered at this estimate, compute mean $\mu$ and standard deviation $\sigma$ of the scores in this region. The z-score is defined as,
$$z = (f^* - \mu)/\sigma$$

Here is a plot of these z-scores for all structures. Bars represent the range of z-scores as the radius of pooling region changes between 20 and 50 pixels.

![zscores](zscores.png)

The structures with low z-scores require attention.

Also, the Hessians are computed.

The third metric is the peak radius. The peak radius $r$ is defined as the shortest distance along any direction away from the peak that the score drops to region mean $\mu$. Assume a quadratic approximation of the score, the peak radius can be expressed using the eigenvalues and eigenvectors of the Hessian matrix $H$:

$$r = \sqrt{2(f^*-\mu)/\lambda_0} v_0$$

where $f^*$ is the peak score, $\mu$ is the pooling region mean, $\lambda_0$ and
$v_0$ are the largest eigenvalue and the corresponding eigenvector of the Hessian matrix $H$.

![peak radius](peak_radius.png)

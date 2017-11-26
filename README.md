<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Optimization-Algorithms

In my recent works, I find some interesting problems about Adam optimizer in sparse cases.
To get a deeper understanding, I read this [overview blog](http://ruder.io/optimizing-gradient-descent/index.html) and corresponding papers and give a presentation about optimization algorithms in APEX Lab.
I want to summarize and share my understandings and confusions about these brilliant works.
I will upload my PPT generally discussing these algorithms, and then update recent studies going deeper in this direction.

## Basic Algorithms (Included in PPT)
This section goes through 2nd order optimization at first to briefly explain ``step size estimation``, and then goes through 1st order optimization which is much more popular in deep learning and other applications involving neural networks. 

### 1st/2nd Order Optimization

#### 1st order optimization: 
(i) The gradient tells us whether the objective is descreasing or increasing at a point, which approximates ``a tangent line`` on the error surface. (ii) A gradient is represented by a ``Jacobian`` Matrix. (iii). The most typical algorithm is ``Gradient Descent``, which updates paramaeters along the steepest descent direction 

![equation](https://latex.codecogs.com/gif.latex?\theta_{k&plus;1}&space;=&space;\theta_k&space;-&space;\eta&space;\nabla_{\theta}&space;J(\theta_k)) 

where $\theta$ denotes the parameter set, $\eta$ denotes the step size, and $J$ represents the objective function.



#### 2nd order optimization: 
(i) Use the 2nd order derivative to optimize the objective, which provides a ``quadratic surface`` touching the curvature of the error surface. (ii) The 2nd order gradient is represented by a ``Hessian`` Matrix. (iii) The most typical algorithm is ``Newton Method``, which estimates a sequence of optima through quadratic curves and gets rid of the step size. ``Newton Method`` can be derived from Tylor Expansion $$J(\theta + \Delta) = J(\theta) + G(\theta) \Delta + \Delta^T H(\theta) \Delta + o(\Delta^2),$$ take derivative and set as 0 to estimate the optimum $$0 = G(\theta) \Delta + \Delta^T H(\theta) \Delta,$$ we get the next estimation of optimum $$\theta_{k+1} = \theta_k - H^{-1}(\theta_k) G(\theta_k),$$ where \\(\eta\\) is replaced by Hassian inverse, be careful that Hassian inverse may change direction.

#### Newton Method 
Newton method guarantees convergence when: (i) \\(\theta_0\\) is close to \\(\theta^*\\), (ii) Hessian of \\(\theta^*\\) is not singular, (iii) Hessian around \\(\theta^*\\) is k-Lipschitz continuous:
$$ \Vert \theta_{k+1} - \theta^\*\Vert = \Vert \theta_k - \theta^\* - h(\theta_k)^{-1} g(\theta_k)\Vert $$ $$ = \Vert h(\theta_k)^{-1}(g(\theta_k) - h(\theta_k)(\theta_k - \theta^\*))\Vert$$ $$= \Vert h(\theta_k^{-1}(g(\theta_k) - g(\theta^\*) - h(\theta_k)(\theta_k - \theta^\*))\Vert$$ $$\le \Vert h(\theta_k^{-1}\Vert * \Vert g(\theta_k) - g(\theta^\*) - h(\theta_k)(\theta_k - \theta^\*))\Vert$$ $$\le \Vert h(\theta_k^{-1}\Vert * k \Vert \theta_k - \theta^\*\Vert^2 \le \frac{k}{\lambda_{min}} \Vert \theta_k - \theta^\*\Vert^2$$
when \\(\frac{k}{\lambda_{min}} < 1\\).

#### Quasi-Newton Method
Because Hessian is difficult to compute, Quasi-Newton methods construct a positive definite symmetric matrix to approximate Hassian (or inverse Hessian) instead according to Quasi-Newton condition $$g_{k+1} - g_k \approx H_{k+1} * (\theta_{k+1} - \theta_k)$$ $$\text{let }s_k = \theta_{k+1} - \theta_k, y_k = g_{k+1} - g_k$$ $$y_k \approx H_{k+1} * s_k, s_k \approx H_{k+1}^{-1} * y_k$$ $$y_k = B_{k+1} * s_k, s_k = D_{k+1} * y_k,$$ thus we can construct \\(B\\) and \\(D\\) matrix via parameter change and gradient change, this leads to Davindon-Fletcher-Powell algorithm (DFP, details in PPT), Broyden-Fletcher-Goldfarb-Shanno algorithm (BFGS, details in PPT), and Limited-memory BFGS (L-BFGS).

#### Why not 2nd order optimization
- Complexity: GD \\(O(n)\\), Quasi-Newton \\(O(n^2)\\), Newton \\(O(n^3)\\), L-BFGS \\(O(kn)\\), where n is the amount of parameters
- (I found in [this post](https://www.zhihu.com/question/53218358) but not sure) Cramer-Rao bound states that generalization error cannot decrease faster than linear in strongly convex problems. From my understanding, 2nd order does not necessarily converge to better solution.
- Robustness: 2nd order methods always have numerical stability issues, etc.

## Further Reading
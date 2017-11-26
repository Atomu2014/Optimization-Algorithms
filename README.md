<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Optimization-Algorithms

In my recent works, I find some interesting problems about Adam optimizer in sparse cases.
To get a deeper understanding, I read this [overview blog](http://ruder.io/optimizing-gradient-descent/index.html) and corresponding papers and give a presentation about optimization algorithms in APEX Lab with the help of ``@Guoxin Sui``.
I want to summarize and share my understandings and confusions about these brilliant works in this repository.
I will upload my PPT generally discussing these algorithms, and then update recent studies going deeper in this direction.

## Basic Algorithms (Included in PPT)
This section goes through 2nd order optimization at first to briefly explain ``step size estimation``, and then goes through 1st order optimization which is much more popular in deep learning and other applications involving neural networks. 

### 1st order optimization: 
- The gradient tells us whether the objective is descreasing or increasing at a point, which approximates ``a tangent line`` on the error surface. 
- A gradient is represented by a ``Jacobian`` Matrix. 
- The most typical algorithm is ``Gradient Descent``, which updates paramaeters along the steepest descent direction 

![equation](https://latex.codecogs.com/gif.latex?%5Ctheta_%7Bk&plus;1%7D%20%3D%20%5Ctheta_k%20-%20%5Ceta%20%5Cnabla_%7B%5Ctheta%7D%20J%28%5Ctheta_k%29)

where \\(\theta\\) denotes the parameter set, \\(\eta\\) denotes the step size, and \\(J\\) represents the objective function.

### 2nd order optimization: 
- Use the 2nd order derivative to optimize the objective, which provides a ``quadratic surface`` touching the curvature of the error surface. 
- The 2nd order gradient is represented by a ``Hessian`` Matrix. 
- The most typical algorithm is ``Newton Method``, which estimates a sequence of optima through quadratic curves and gets rid of the step size. ``Newton Method`` can be derived from Tylor Expansion 

![equation](https://latex.codecogs.com/gif.latex?J%28%5Ctheta%20&plus;%20%5CDelta%29%20%3D%20J%28%5Ctheta%29%20&plus;%20G%28%5Ctheta%29%20%5CDelta%20&plus;%20%5CDelta%5ET%20H%28%5Ctheta%29%20%5CDelta%20&plus;%20o%28%5CDelta%5E2%29)

take derivative and set as 0 to estimate the optimum 

![equation](https://latex.codecogs.com/gif.latex?0%20%3D%20G%28%5Ctheta%29%20%5CDelta%20&plus;%20%5CDelta%5ET%20H%28%5Ctheta%29%20%5CDelta)

we get the next estimation of optimum 

![equation](https://latex.codecogs.com/gif.latex?%5Ctheta_%7Bk&plus;1%7D%20%3D%20%5Ctheta_k%20-%20H%5E%7B-1%7D%28%5Ctheta_k%29%20G%28%5Ctheta_k%29)

where \\(\eta\\) is replaced by Hassian inverse, be careful that Hassian inverse may change direction.

### Newton Method 
Newton method guarantees convergence when: (i) \\(\theta_0\\) is close to \\(\theta^*\\), (ii) Hessian of \\(\theta^*\\) is not singular, (iii) Hessian around \\(\theta^*\\) is k-Lipschitz continuous:

![equation](https://latex.codecogs.com/gif.latex?%5CVert%20%5Ctheta_%7Bk&plus;1%7D%20-%20%5Ctheta%5E*%5CVert%20%5C%5C%20%26%20%3D%20%5CVert%20%5Ctheta_k%20-%20%5Ctheta%5E*%20-%20h%28%5Ctheta_k%29%5E%7B-1%7D%20g%28%5Ctheta_k%29%5CVert%20%5Cnonumber%20%5C%5C%20%26%20%3D%20%5CVert%20h%28%5Ctheta_k%29%5E%7B-1%7D%28g%28%5Ctheta_k%29%20-%20h%28%5Ctheta_k%29%28%5Ctheta_k%20-%20%5Ctheta%5E*%29%29%5CVert%20%5C%5C%20%26%20%3D%20%5CVert%20h%28%5Ctheta_k%5E%7B-1%7D%28g%28%5Ctheta_k%29%20-%20g%28%5Ctheta%5E*%29%20-%20h%28%5Ctheta_k%29%28%5Ctheta_k%20-%20%5Ctheta%5E*%29%29%5CVert%20%5C%5C%20%26%20%5Cle%20%5CVert%20h%28%5Ctheta_k%5E%7B-1%7D%5CVert%20*%20%5CVert%20g%28%5Ctheta_k%29%20-%20g%28%5Ctheta%5E*%29%20-%20h%28%5Ctheta_k%29%28%5Ctheta_k%20-%20%5Ctheta%5E*%29%29%5CVert%20%5C%5C%20%26%20%5Cle%20%5CVert%20h%28%5Ctheta_k%5E%7B-1%7D%5CVert%20*%20k%20%5CVert%20%5Ctheta_k%20-%20%5Ctheta%5E*%5CVert%5E2%20%5C%5C%20%26%20%5Cle%20k%20/%20%5Clambda_%7Bmin%7D%20*%20%5CVert%20%5Ctheta_k%20-%20%5Ctheta%5E*%5CVert%5E2)

when \\(\frac{k}{\lambda_{min}} < 1\\).

### Quasi-Newton Method
Because Hessian is difficult to compute, Quasi-Newton methods construct a positive definite symmetric matrix to approximate Hassian (or inverse Hessian) instead according to Quasi-Newton condition 

![equation](https://latex.codecogs.com/gif.latex?%5C%5Cg_%7Bk&plus;1%7D%20-%20g_k%20%5Capprox%20H_%7Bk&plus;1%7D%20*%20%28%5Ctheta_%7Bk&plus;1%7D%20-%20%5Ctheta_k%29%20%5C%5C%20%5Ctext%7Blet%20%7Ds_k%20%3D%20%5Ctheta_%7Bk&plus;1%7D%20-%20%5Ctheta_k%2C%20y_k%20%3D%20g_%7Bk&plus;1%7D%20-%20g_k%20%5C%5C%20y_k%20%5Capprox%20H_%7Bk&plus;1%7D%20*%20s_k%2C%20s_k%20%5Capprox%20H_%7Bk&plus;1%7D%5E%7B-1%7D%20*%20y_k%20%5C%5C%20y_k%20%3D%20B_%7Bk&plus;1%7D%20*%20s_k%2C%20s_k%20%3D%20D_%7Bk&plus;1%7D%20*%20y_k%20%5C%5C)

thus we can construct \\(B\\) and \\(D\\) matrix via parameter change and gradient change, this leads to Davindon-Fletcher-Powell algorithm (DFP, details in PPT), Broyden-Fletcher-Goldfarb-Shanno algorithm (BFGS, details in PPT), and Limited-memory BFGS (L-BFGS).

### Challenges of 2nd order optimization
- Complexity: GD \\(O(n)\\), Quasi-Newton \\(O(n^2)\\), Newton \\(O(n^3)\\), L-BFGS \\(O(kn)\\), where n is the amount of parameters
- (I found in [this Chinese post](https://www.zhihu.com/question/53218358) but not sure) Cramer-Rao bound states that generalization error cannot decrease faster than linear in strongly convex problems. From my understanding, 2nd order does not necessarily converge to better solution.
- Robustness: 2nd order methods always have numerical stability issues, etc.

### Challenges of 1st order optimization
- How to escape from local minima and saddle points
- How to decide learning rate (schedule)

Momentum-based algorithms are proposed to solve the 1st problem.
Adaptive learning rate algorithms are proposed to solve the 2nd problem.

### Momentum
``On the momentum term in gradient descent learning algorithms, Ning Qian, 1999``

![equation](https://latex.codecogs.com/gif.latex?%5C%5Cv_t%20%3D%20%5Cgamma%20v_%7Bt-1%7D%20&plus;%20%5Ceta%20%5Cnabla_%5Ctheta%20J%28%5Ctheta%29%29%20%5C%5C%5Ctheta%20%3D%20%5Ctheta%20-%20v_t)

Putting the learning rate \\(\eta\\) in eq.(1) or eq.(2) has no difference.
Momentum is proposed to accelerate training in plateaus, but it does not follow the gradient direction.
The potential cost of speed-up is performance, and this problem arises in other momentum-based algorithms, e.g., Nesterov, Adam.

### Nesterov Accelerated Gradient

![equation](https://latex.codecogs.com/gif.latex?%5C%5Cv_t%20%3D%20%5Cgamma%20v_%7Bt-1%7D%20&plus;%20%5Ceta%20%5Cnabla_%5Ctheta%20J%28%5Ctheta%20-%20%5Cgamma%20v_%7Bt-1%7D%29%29%20%5C%5C%5Ctheta%20%3D%20%5Ctheta%20-%20v_t)

Nesterov differs from momentum by replacing the last step gradient with a predicted gradient of the next step.
The prediction is more precise when the cumulated term is larger than the real gradient, as illustrated in [Hinton's lecture 6c](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).
Comparing to Momentum, Nesterov predicts a closer gradient of the next step, which approximates the local curvature better and converges faster than momentum.
[This Chinese post](https://zhuanlan.zhihu.com/p/22810533) shows the equivalence of Momentum and Nesterov.

### Adagrad

``Adaptive Subgradient Methods for Online Learning and Stochastic Optimization, John Duchi, et. al. 2011``

![equation](https://latex.codecogs.com/gif.latex?%5C%5Cg_%7Bt%2Ci%7D%20%3D%20%5Cnabla_%5Ctheta%20J%28%5Ctheta_i%29%29%20%5C%5CG_%7Bt%2Cii%7D%20%3D%20%5Csum_%7B%5Ctau%3D1%7D%5Et%20g_%7B%5Ctau%2Ci%7D%5E2%20%5C%5C%5Ctheta_%7Bt&plus;1%7D%20%3D%20%5Ctheta_%7Bt%7D%20-%20%5Cfrac%7B%5Ceta%7D%7B%5Csqrt%7BG_t%20&plus;%20%5Cepsilon%7D%7D%20%5Codot%20g_t)

Different from Momentum and Nesterov, Adagrad is an adaptive algorithm because it estimates learning rates for different parameters using L2 norm of gradient sequence. This idea comes from the generalization of Gradient Descent to Mahalanobis space, which is formed by historical gradients.

![equation](https://latex.codecogs.com/gif.latex?%5C%5C%5Ctheta_%7Bt&plus;1%7D%20%3D%20%5Carg%5Cmin%20%5CVert%20%5Ctheta%20-%20%28%5Ctheta_t%20-%20%5Ceta%20g_t%29%29%5CVert%20%5C%5C%5Ctext%7Bchange%20L2-norm%20to%20Mahalanobis%20norm%2C%7D%20%5C%5C%5Ctext%7Bdefine%20%7DG_t%20%3D%20%5Csum_%7B%5Ctau%3D1%7D%5Et%20g_%5Ctau%20g_%5Ctau%5ET%20%5C%5Cx_%7Bt&plus;1%7D%20%3D%20%5Carg%5Cmin%20%5CVert%20%5Ctheta%20-%20%28%5Ctheta_t%20-%20%5Ceta%20G_t%5E%7B-1/2%7Dg_t%29%29%20%5CVert%20%5C%5C%5Ctext%7Bfor%20easy%20computation%2C%20replace%20%7D%20G_t%5E%7B1/2%7D%20%5Ctext%7B%20with%20%7D%20diag%28G_t%29%5E%7B1/2%7D)

Adagrad is a very important adaptive algorithm and can adjust to features with different frequency. However, the cumulated sum may result in too early stop in training even though reasonable gradients are provided.

### Adadelta

``Adaldelta: An Adaptive Learning Rate Method, Matthew D. Zeiler, 2012``

Adadelta improves Adagrad by replacing the sum of squared gradients with the expectation, for simplicity, Adadelta uses an exponentially decaying average of the squared gradients

![equation](https://latex.codecogs.com/gif.latex?%5C%5CE%5Bg%5E2%5D_t%20%3D%20%5Cgamma%20E%5Bg%5E2%5D_%7Bt-1%7D%20&plus;%20%281-%5Cgamma%29%20g_t%5E2%20%5C%5C%5Ctheta_%7Bt&plus;1%7D%20%3D%20%5Ctheta_t%20-%20%5Cfrac%7B%5Ceta%7D%7B%5Csqrt%7BE%5Bg%5E2%5D_t%20&plus;%20%5Cepsilon%7D%7D%20g_t%20%3D%20%5Ctheta_t%20-%20%5Cfrac%7B%5Ceta%7D%7BRMS%28g%29_t%7D%20g_t)

where \\(\epsilon\\) is a small number for numerical stability.
The exponentially decaying average can also be regarded as the discounted sum in a sliding window, which can avoid the infinite sum of Adagrad and reflect local training dynamics. Adadelta also considers the weakness of Adagrad comes from the sensitivity to learning rate. Thus Adadelta further replaces the constant learning rate term with an estimated step size.

![equation](https://latex.codecogs.com/gif.latex?%5CDelta%20%5Ctheta%20%3D%20-%20%5Cfrac%7BRMS%28%5CDelta%20%5Ctheta%29_%7Bt-1%7D%7D%7BRMS%28g%29_t%7D%20g_t)

The author explains this idea with ``unit correction``, a very relevant work is ``No more pesky learning rates, T. Schaul, S. Zhang, and Y. LeCun, 2012``, where the windowed expectations are incorporating the diagonal Hessian to better estimate Hessian.

![equation](https://latex.codecogs.com/gif.latex?%5CDelta%20%5Ctheta%20%3D%20-%20%5Cfrac%7B1%7D%7B%7Cdiag%28H_t%29%7C%7D%20%5Cfrac%7BE%5Bg_%7Bt-w%3At%7D%5D%5E2%7D%7BE%5Bg%5E2_%7Bt-w%3At%7D%5D%7D%20g_t)

I think Adadelta does not provide sufficient proof of its step size estimation and ``unit correction`` is too simple to generalize to other cases. Besides, the author regards the windowed sum and the expectation as the same, but they are slightly different actually. This is because of the 0 initialization of the expectation, which yields a biased estimation in the early stage of training.

### RMSProp

RMSProp is introduced in [Hinton's lecture 6e](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf), and is a simpler version of Adadelta.

![equation](https://latex.codecogs.com/gif.latex?%5C%5CE%5Bg%5E2%5D_t%20%3D%20%5Cgamma%20E%5Bg%5E2%5D_%7Bt-1%7D%20&plus;%20%281-%5Cgamma%29%20g_t%5E2%20%5C%5C%5Ctheta_%7Bt&plus;1%7D%20%3D%20%5Ctheta_t%20-%20%5Cfrac%7B%5Ceta%7D%7BRMS%28g%29_t%7D%20g_t)

where \\(\gamma = 0.9\\), \\(\eta = 0.001\\) is the default setting.
RMSProp is also popular in practice, a possible reason is that RMSProp is simpler, another possible reason is that the step size estimation of Adadelta does not always work well.

## Further Reading
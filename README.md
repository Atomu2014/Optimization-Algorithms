<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Optimization-Algorithms

In my recent works, I find some interesting problems about Adam optimizer in sparse cases.
To get a deeper understanding, I read this [overview blog](http://ruder.io/optimizing-gradient-descent/index.html) and corresponding papers and give a presentation about optimization algorithms in APEX Lab with the help of ``@Guoxin Sui``.
I want to summarize and share my understandings and confusions about these brilliant works in this repository.
I will upload my PPT generally discussing these algorithms, and then update recent studies going deeper in this direction.

## Basic Algorithms (Included in PPT)
This section goes through 2nd order optimization at first to briefly explain ``step size estimation``, and then goes through 1st order optimization which is much more popular in deep learning and other applications involving neural networks. 

### 1st/2nd Order Optimization

#### 1st order optimization: 
(i) The gradient tells us whether the objective is descreasing or increasing at a point, which approximates ``a tangent line`` on the error surface. (ii) A gradient is represented by a ``Jacobian`` Matrix. (iii). The most typical algorithm is ``Gradient Descent``, which updates paramaeters along the steepest descent direction 

![equation](https://latex.codecogs.com/gif.latex?%5Ctheta_%7Bk&plus;1%7D%20%3D%20%5Ctheta_k%20-%20%5Ceta%20%5Cnabla_%7B%5Ctheta%7D%20J%28%5Ctheta_k%29)

where \\(\theta\\) denotes the parameter set, \\(\eta\\) denotes the step size, and \\(J\\) represents the objective function.

#### 2nd order optimization: 
(i) Use the 2nd order derivative to optimize the objective, which provides a ``quadratic surface`` touching the curvature of the error surface. (ii) The 2nd order gradient is represented by a ``Hessian`` Matrix. (iii) The most typical algorithm is ``Newton Method``, which estimates a sequence of optima through quadratic curves and gets rid of the step size. ``Newton Method`` can be derived from Tylor Expansion 

![equation](https://latex.codecogs.com/gif.latex?J%28%5Ctheta%20&plus;%20%5CDelta%29%20%3D%20J%28%5Ctheta%29%20&plus;%20G%28%5Ctheta%29%20%5CDelta%20&plus;%20%5CDelta%5ET%20H%28%5Ctheta%29%20%5CDelta%20&plus;%20o%28%5CDelta%5E2%29)

take derivative and set as 0 to estimate the optimum 

![equation](https://latex.codecogs.com/gif.latex?0%20%3D%20G%28%5Ctheta%29%20%5CDelta%20&plus;%20%5CDelta%5ET%20H%28%5Ctheta%29%20%5CDelta)

we get the next estimation of optimum 

![equation](https://latex.codecogs.com/gif.latex?%5Ctheta_%7Bk&plus;1%7D%20%3D%20%5Ctheta_k%20-%20H%5E%7B-1%7D%28%5Ctheta_k%29%20G%28%5Ctheta_k%29)

where \\(\eta\\) is replaced by Hassian inverse, be careful that Hassian inverse may change direction.

#### Newton Method 
Newton method guarantees convergence when: (i) \\(\theta_0\\) is close to \\(\theta^*\\), (ii) Hessian of \\(\theta^*\\) is not singular, (iii) Hessian around \\(\theta^*\\) is k-Lipschitz continuous:

![equation](https://latex.codecogs.com/gif.latex?%5CVert%20%5Ctheta_%7Bk&plus;1%7D%20-%20%5Ctheta%5E*%5CVert%20%5C%5C%20%26%20%3D%20%5CVert%20%5Ctheta_k%20-%20%5Ctheta%5E*%20-%20h%28%5Ctheta_k%29%5E%7B-1%7D%20g%28%5Ctheta_k%29%5CVert%20%5Cnonumber%20%5C%5C%20%26%20%3D%20%5CVert%20h%28%5Ctheta_k%29%5E%7B-1%7D%28g%28%5Ctheta_k%29%20-%20h%28%5Ctheta_k%29%28%5Ctheta_k%20-%20%5Ctheta%5E*%29%29%5CVert%20%5C%5C%20%26%20%3D%20%5CVert%20h%28%5Ctheta_k%5E%7B-1%7D%28g%28%5Ctheta_k%29%20-%20g%28%5Ctheta%5E*%29%20-%20h%28%5Ctheta_k%29%28%5Ctheta_k%20-%20%5Ctheta%5E*%29%29%5CVert%20%5C%5C%20%26%20%5Cle%20%5CVert%20h%28%5Ctheta_k%5E%7B-1%7D%5CVert%20*%20%5CVert%20g%28%5Ctheta_k%29%20-%20g%28%5Ctheta%5E*%29%20-%20h%28%5Ctheta_k%29%28%5Ctheta_k%20-%20%5Ctheta%5E*%29%29%5CVert%20%5C%5C%20%26%20%5Cle%20%5CVert%20h%28%5Ctheta_k%5E%7B-1%7D%5CVert%20*%20k%20%5CVert%20%5Ctheta_k%20-%20%5Ctheta%5E*%5CVert%5E2%20%5C%5C%20%26%20%5Cle%20k%20/%20%5Clambda_%7Bmin%7D%20*%20%5CVert%20%5Ctheta_k%20-%20%5Ctheta%5E*%5CVert%5E2)

when \\(\frac{k}{\lambda_{min}} < 1\\).

#### Quasi-Newton Method
Because Hessian is difficult to compute, Quasi-Newton methods construct a positive definite symmetric matrix to approximate Hassian (or inverse Hessian) instead according to Quasi-Newton condition 

![equation](https://latex.codecogs.com/gif.latex?%5C%5Cg_%7Bk&plus;1%7D%20-%20g_k%20%5Capprox%20H_%7Bk&plus;1%7D%20*%20%28%5Ctheta_%7Bk&plus;1%7D%20-%20%5Ctheta_k%29%20%5C%5C%20%5Ctext%7Blet%20%7Ds_k%20%3D%20%5Ctheta_%7Bk&plus;1%7D%20-%20%5Ctheta_k%2C%20y_k%20%3D%20g_%7Bk&plus;1%7D%20-%20g_k%20%5C%5C%20y_k%20%5Capprox%20H_%7Bk&plus;1%7D%20*%20s_k%2C%20s_k%20%5Capprox%20H_%7Bk&plus;1%7D%5E%7B-1%7D%20*%20y_k%20%5C%5C%20y_k%20%3D%20B_%7Bk&plus;1%7D%20*%20s_k%2C%20s_k%20%3D%20D_%7Bk&plus;1%7D%20*%20y_k%20%5C%5C)

thus we can construct \\(B\\) and \\(D\\) matrix via parameter change and gradient change, this leads to Davindon-Fletcher-Powell algorithm (DFP, details in PPT), Broyden-Fletcher-Goldfarb-Shanno algorithm (BFGS, details in PPT), and Limited-memory BFGS (L-BFGS).

#### Why not 2nd order optimization
- Complexity: GD \\(O(n)\\), Quasi-Newton \\(O(n^2)\\), Newton \\(O(n^3)\\), L-BFGS \\(O(kn)\\), where n is the amount of parameters
- (I found in [this post](https://www.zhihu.com/question/53218358) but not sure) Cramer-Rao bound states that generalization error cannot decrease faster than linear in strongly convex problems. From my understanding, 2nd order does not necessarily converge to better solution.
- Robustness: 2nd order methods always have numerical stability issues, etc.

## Further Reading
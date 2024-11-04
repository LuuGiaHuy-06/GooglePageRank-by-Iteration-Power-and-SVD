# GooglePageRank-by-Iteration-Power-and-SVD
Compare the speed of the Google Page Rank problem between using SVD in NumPy and power iteration method.

# Singular Value Decomposition (SVD)

### 1. Eigenvalue and Eigenvector
The number $\lambda$ is an eigenvalue of $A$ if and only if $A-\lambda I$ is singular. For each eigenvalue $\lambda$ solve $A-\lambda I = 0$ to find an eigenvector $x$.


### 2. Diagonalizing a Matrix
Suppose $n\times n$ matrix A has $n$ *linear independent eigenvectors* $x_{1}, x_{2},..., x_{n}$. Let $X$ is an eigenvector matrix then

$X^{-1}AX = \Lambda$ because $AX = [x_{1}\lambda_{1} ... x_{n}\lambda_{n}] = X\Lambda$

Notice: $X$ is invertible because we assumped that A has $n$ linear independent eigenvectors. Without this condition, a matrix can't be diagonalize.

### 3. Symmetric Matrices
1. Has only *real eigenvalues*
2. Eigenvectors can be chosen *orthonormal*

**Spectral Theorem** Every symmetric matrix S has the factorization $S = Q\Lambda Q^{T}$ with real eigenvalues in $\Lambda$ and orthonormal eigenvectors in $Col(Q)$.
**Proof 1**
Suppose $\lambda = a+ib$ we have: 

$Sx=\lambda x \Rightarrow \bar x^{T} Sx=\bar x^{T} \lambda x$ and $S \bar x=\bar\lambda \bar x \Rightarrow \bar x^{T} S=\bar x^{T} \bar\lambda \Rightarrow \bar x^{T} S x=\bar x^{T} \bar\lambda x$

Since $\bar x^{T} x \geq 0 \Rightarrow \lambda = \bar \lambda = a$ is real.

**Proof 2**
Suppose $S x = \lambda_{1} x$, $S y = \lambda_{2} y$ and $\lambda_{1} \neq \lambda_{2}$ we have:

$(\lambda_{1} x)^{T} y = (S x)^{T} y = x^{T} S y = x^{T} (\lambda_{2} y)$ ($S$ is symmetric).

Therefor $x^{T} y = 0 $. With full linear independent eigenvectors $Q$ is orthogonal.


### 4. SVD

Every $m \times n$ matrix $A$ has the factorization $A = U \Lambda V^{T}$

with $A A{T} = U \Lambda_{U} U^{T}$ and $A{T} A = V \Lambda_{V} V^{T} $ and $A A^{T}$ and $A^{T} A$ have same non-zero eigenvalues.

**Proof**
Suppose $\lambda_{0}, X_{0}$ are eigenvalue, eigenvector of  $AA^{T}$ 

$AA^{T}X_{0} = \lambda_{0}X_{0}$ $\Leftrightarrow$ $A^{T}A(A^{T}X_{0}) = \lambda_{0}(A^{T}X_{0})$ with $\lambda_{0} \neq 0$: $\lambda_{0}$ is also the eigenvalue of $A^{T}A$


# Power Iteration method
The algorithm with return the greatest $\lambda$ (in absolute value) and a non zero vector $v$ which $A v = \lambda v$.

Start with $random b_{0}$, at every iteration:

$b_{k+1} = \frac{A b_{k}}{||A b_{k}||} = \frac{A^{k+1} b_{0}}{||A^{k+1} b_{0}||}$

**Proof**
Choose $b_{0} = \sum{c_{i} v_{i}}$. Then

$A^{k} b_{0} = \sum{c_{i} A^{k} v_{i}} = \sum{c_{i} \lambda_{i}^{k} v_{i}} = c_{1} \lambda_{1}^{k} \bullet \sum{\left({ \frac{c_{i}}{c_{1}} \left( \frac{\lambda_{i}}{\lambda_{1}} \right)^{k} v_{i} } \right)} \rightarrow c_{1} \lambda_{1}^{k} v_{1}$ 

when $|\frac{\lambda_{i}}{\lambda_{1}}| < 1$

# Complexity 

## Google Matrix $G$
...

## Use ```numpy.linalg.eig```
Depends on size of $G$, typically costing $O\left(n^{3}\right)$ for dense matrix, but *Google Page Rank matrix* is generally sparse.

## Use power iteration
Per iteration cost: $O\left(n^{2}\right)$ and usually stops before $k$ reachs $n$ so the cost is $O\left(k \bullet n^{2}\right)$

But the algorithm usually works efficient for sparse matrix, and return the domination eigenvalue/eigenvector (if it exists).


# References
[1] Gilbert Strang, *Introduction to Linear Algebra*; 5th edition.

[2] Wikipedia, *https://en.wikipedia.org/wiki/Power_iteration*.




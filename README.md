# GooglePageRank-by-Iteration-Power-and-SVD
Compare the speed of the Google Page Rank problem between using SVD in NumPy and power iteration method.

# Singular Value Decomposition (SVD)
With a real $m \times n$ matrix $A$ we always have: $A*A^{T}$ and $A^{T}*A$ have same non-zero eigenvalues.

Prove: 
Suppose $\lambda_{0}, X_{0}$ are eigenvalue, eigenvector of  $A*A^{T}$
$A*A^{T}*X_{0} = \lambda_{0}*X_{0} \Leftrightarrow A^{T}*A*\left(A^{T}*X_{0}\right) = \lambda_{0}\left(A^{T}*X_{0}\right)$
with $\lambda_{0} \neq 0$:
$\lambda_{0}$ is also the eigenvalue of $A^{T}*A$

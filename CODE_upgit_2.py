import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
#Clean dataset
df = pd.read_csv('2019.csv')            
country_names = df['Country or region']            
df.drop(['Overall rank', 'Country or region'], axis=1, inplace=True)       
scaler = StandardScaler()
X = scaler.fit_transform(df)
# Power
cov_matrix = np.cov(X, rowvar=False)  # S = 1/(m-1) * X.T @ X
def power_iteration(A, num_iterations=1000):
    b_k = np.random.rand(A.shape[1])
    for _ in range(num_iterations):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    return b_k
def find_eigenvalue(b, A):
    return np.dot(b.T, A @ b)
# Find the 1st and 2nd principal component
vec1 = power_iteration(cov_matrix)
eigval_1 = find_eigenvalue(vec1, cov_matrix)
cov_matrix_new = cov_matrix - eigval_1 * np.outer(vec1, vec1)
vec2 = power_iteration(cov_matrix_new)
# Gram-Schmidt process
vec2 = vec2 - np.dot(vec1, vec2) * vec1
vec2 /= np.linalg.norm(vec2)
eigvecs = np.vstack((vec1, vec2)).T
# Find PCA components
principal_components = X @ eigvecs
happiness_index = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
happiness_index['Country'] = country_names
#plot
plt.figure(figsize=(10, 8))
plt.scatter(happiness_index['PC1'], happiness_index['PC2'], alpha=0.7, c='red')
for i, country in enumerate(happiness_index['Country']):
    plt.annotate(country, (happiness_index['PC1'][i], happiness_index['PC2'][i]),
                 fontsize=8, alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Happiness Index')
plt.grid(True)
plt.show()

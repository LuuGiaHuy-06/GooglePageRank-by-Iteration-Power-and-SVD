import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import time
import random

"""
To compute SVD of the square matrix A (n x n) in python, we can use numpy.linalg.eig(A)
Calculating an (n x n) matrix has a computational complexity of O(n^3) for dense matrices
Use power iteration method with k times of iterations has a computational complexity of O(n^2 * k)
Typically, k < n if you're only interested in the largest eigenvalue and corresponding eigenvector.

SUMMARIZE:
For finding ONLY the largest eigenvector, power iteration is almost always more efficient.
Application: GooglePageRank
"""



#create Stochastic matrix S: square matrix n*n
#nay la vi du trong sach thay V
SS = np.array([
    [0, 1/3, 0, 0, 0, 0, 0],
    [1/3, 0, 1, 1/2, 0, 0, 0],
    [1/3, 1/3, 0, 1/2, 0, 0, 0],
    [1/3, 1/3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1/2, 0, 0],
    [0, 0, 0, 0, 1/2, 1, 0]
             ])

#because the Stochastic matrix S is the Markov matrix, we create one below:
def Markov(transitions):
    n = 1+ max(transitions) #number of states
    M = [[0]*n for _ in range(n)]
    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1
    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M
t=[]
n = 1000
for i in range(n):
    t.append(random.randint(1, n))
S = np.transpose(np.array(Markov(t)))


#Personalization vectors v>0 and ||v|| = 1
n = np.sqrt(np.size(S))
V = np.full_like(S, 1/n)
#damping factor 0<alpha<1, e.g. alpha =.85
alpha = 0.85


"""
G = alpha*S + (1-alpha)*V
properties: 
- G has eigenvalue 1 (stochastic)
- spectral radius 1 unique (primitive)


Unique left eigenvector: G*b = b, b>0 and ||b|| = 1
i-th entry of b: PageRank of page i
PageRank = largest left eigenvector of G
"""
#Google Matrix 
G = alpha * S + (1-alpha) * V


class iteration_power():
    def vector(A, num_iterations = 10000): #set the default number of iteration equals to 10 000
        b_k = np.random.rand(A.shape[1])   #generate random vector b

        for _ in range(num_iterations):            #b_{k+1} = (b_{k}*A)/ ||b_{k}*A||
            b_k1 = np.dot(A, b_k)                  #b_{k}*A

            b_k1_norm = np.linalg.norm(b_k1)       #||b_{k}*A||

            b_k = b_k1 / b_k1_norm
        return b_k      

    def find_eigenvalue(b, A):            # we know that Ab = λb <=> λ = ( b_k^-1 * A * b_k )/ ||b_k||^2
        lamb = np.dot( np.transpose(b) , np.dot(A , b) ) / np.linalg.norm(b)**2
        return lamb

    def new_matrix(A, lamb, b):
        b = np.array(b)                  #b become matrix
        b = b.reshape(-1, 1)
        b_T = np.transpose(b)
        newA = A - (lamb * b @ b_T )/ (np.linalg.norm(b)**2)  # B = A - λ*(b_k * b_k^T) / ||b_k||^2
        return newA 
    #in case we want to find more than one dominate eigenvalue 
    def list_of_eigenvalues(A, number):    #return a list of dominate eigenvalues of A
        v_dict = []
        i = 0
        while i < number:
            i += 1
            v = iteration_power.vector(A)
            lamb = iteration_power.find_eigenvalue(v, A)
            if lamb <= 0 or lamb < 0.01:
                break
            A = iteration_power.new_matrix(A, lamb, v)
            v_dict.append(float(round(lamb, 4)))
        return v_dict
    #list of them
    def list_of_vectors(A, number):     #return a list of vectors b
        v_dict = []
        i = 0
        while i < number:
            i += 1
            v = iteration_power.vector(A)
            lamb = iteration_power.find_eigenvalue(v, A)
            if lamb <= 0 or lamb < 0.01:
                break
            A = iteration_power.new_matrix(A, lamb, v)
            v_dict.append(np.round(v, 4))
        return v_dict

#print(iteration_power.list_of_eigenvalues(G, 1))


start = time.time()
q, w = np.linalg.eig(G)
end = time.time()
print("This is direct way: \n" ,end - start)

start = time.time()
v = iteration_power.vector(G, 40) #the number of iterations depend on your choice.

end = time.time()
print("This is iteration power way: \n",end - start)

#in case want to find the similar between two answers
#print(np.round(w, 4))
#print(np.round(v, 4))

import numpy as np
import time
import random
import scipy
# Markov matrix
random.seed(0)
def Markov(transitions):
    N = 1+max(transitions)
    M = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                M[i][j] = random.random()
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
V = np.full_like(S, 1/n)
alpha = 0.85
# Google Matrix 
G = alpha * S + (1-alpha) * V

def vector(A, num_iterations = 1000): #set the default number of iteration equals to 1000
    b_k = np.random.rand(A.shape[1])   #generate random vector b

    for _ in range(num_iterations):            #b_{k+1} = (b_{k}*A)/ ||b_{k}*A||
        b_k1 = np.dot(A, b_k)                  #b_{k}*A

        b_k1_norm = np.linalg.norm(b_k1)       #||b_{k}*A||

        b_k = b_k1 / b_k1_norm
        #print(np.linalg.norm(b_k)-b_k1_norm)
        if (np.linalg.norm(b_k) - b_k1_norm) < (n*10**(-8)) and _ > 3:   #when || b_{k+1} - b_{k} || < τ where τ =~ 10^-4, 10^-6, 10^-8, ...
            break
    return b_k      

#time
start1 = time.time()
q, w = np.linalg.eig(G)
end1 = time.time()
print("This is np.linalg.eig way: \n" ,end1 - start1)
start2 = time.time()
v = vector(G) #the number of iterations depend on your choice. (default set = 1000)
end2 = time.time()
print("This is iteration power way: \n",end2 - start2)

"""
start4 = time.time()
val2 = np.linalg.eigvals(G)
end4 = time.time()
start5 = time.time()
val3 = scipy.linalg.eig(G)
end5 = time.time()
print("4: ", end4- start4, "5: ", end5-start5)
"""


#in case want to find the similar between two answers
"""
print("part of power iteration result \n",np.round(v, 4))
w = np.transpose(w)
print("part of np.linalg.eig result\n",np.round(w[:1], 4))
"""

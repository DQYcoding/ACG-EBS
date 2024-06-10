import numpy as np

def eucliDist(A,B):
    return np.sqrt(sum(np.power((A - B), 2)))

def eucli_pointoset(sample, matrix):
    N = np.size(matrix, 0)    
    A = sample.reshape(-1)
    if np.size(A) != 16:
        c = 1
    distance = np.ones((N, 1))
    for i in range(N):
        B = matrix[i, :].reshape(-1)
        distance[i] = eucliDist(A, B)
    return np.min(distance)
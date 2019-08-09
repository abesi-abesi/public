import numpy as np
import time
from numpy.random import rand

N = 150

matA = np.array(rand(N,N))
matB = np.array(rand(N,N))
matC = np.array([[0] * N for _ in range(N)])


start = time.time()

for i in range(N):
    for j in range(N):
        for  k in range(N):
            matC[i][j] = matA[i][k] * matB[k][j]

print("Pythonによる計算結果：%.2f[sec]" % float(time.time() - start))

start = time.time()

matC = np.dot(matA,matB)

print("numpyによる計算結果：%.2f[sec]" % float(time.time() - start))
import numpy as np

test = np.load('./edge_weight.npy')
print(test)

a = np.array([[1,2,1],[2,3,2],[3,4,3],[4,5,4]])
b = np.array([[1,1,0,0],[1,1,0,0],[0,1,1,1],[0,0,1,1]])
print(a)
print(b)

c = np.matmul(b,a)
print(c)
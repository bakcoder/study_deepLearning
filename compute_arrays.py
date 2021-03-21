import numpy as np
from study_deepLearning.activation_function import sigmoid_function, identity

# 1차원 배열 = 벡터
A = np.array([1,2,3,4]) # 벡터
print(A)

print(np.ndim(A)) # 1 // get dimension

print(A.shape) # (4,)

print(A.shape[0]) # 4

# 2차원 배열 = 행렬
B = np.array([[1,2] , [3,4], [5,6]])
print(B)
print(np.ndim(B))
print(B.shape)

# 행렬의 곱

A = np.array([[1,2], [3,4]])
print(A.shape)
B = np.array([[5,6], [7,8]])

print(np.dot(A,B))

X = np.array([1,2])
print(X.shape)

W = np.array([[1,3,5],[2,4,6]])
print(W)
print(W.shape)
Y = np.dot(X, W)
print(Y)

def init_network():
    network = {}
    network['W1'] = np.array([[.1, .3, .5], [.2, .4, .6]])
    network['b1'] = np.array([.1, .2, .3])
    network['W2'] = np.array([[.1, .4], [.2, .5], [.3, .6]])
    network['b2'] = np.array([.1, .2])
    network['W3'] = np.array([[.1, .3], [.2, .4]])
    network['b3'] = np.array([.1, .2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid_function(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid_function(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity(a3)

    return y

network = init_network()
x = np.array([1., .5])
y = forward(network, x)
print(y)
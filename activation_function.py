import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def step_function(x):
    if x>0:
        return 1
    else:
        return 0

def step_function_for_numpy(x):
    #y = x>0
    #return y.astype(np.int)
    return np.array(x>0, dtype=np.int)

result = step_function_for_numpy(np.array([1, 0.5, 0.3, 0, -1]))
print(result)


# 계단함수 그래프 그리기

x = np.arange(-5.0, 5.0, 0.1)
y = step_function_for_numpy(x)
# plt.plot(x,y)
# plt.ylim(-0.1, 1.1)
# plt.show()

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

a = np.array([-1.0, 1.0, 2.0])
b = sigmoid_function(x)
print(b)

test = np.array([1.0, 2.0, 3.0]) # 브로드캐스트 테스트
print("1 + test : ", 1+test)
print("1 / test : ", 1/test)

y = sigmoid_function(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()

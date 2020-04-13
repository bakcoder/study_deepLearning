# 퍼셉트론(perceptron) 알고리즘
# 1957년 프랑크 로젠블라트가 고안한 알고리즘
# 신경망(딥러닝)의 기원이 되는 알고리즘

# 다수의 신호를 입력받아 하나의 신호를 출력
# 퍼셉트론 신호은 흐른다 / 안흐른다(1 or 0) => 뉴런 활성화 / 비활성화

# 단순 논리회로
# 1.  AND 게이트

def AND1(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    temp = x1*w1 + x2*w2
    if temp <= theta:
        return 0
    elif temp > theta:
        return 1

# print(AND(0, 0), AND(0,1), AND(0,1), AND(1,1))


import numpy as np
x = np.array([0, 1]) # 입력
w = np.array([.5, .5]) # 가중치
b = -.7

# print("w*x >>> ", w*x)
# print("np.sum(w*x) >>> ", np.sum(w*x))
# print("np.sum(w*x) + b >>> ", np.sum(w*x) + b)

def AND2(x1, x2):
    x = np.array([x1, x2])
    w = np.array([.5, .5])
    b = -.7
    temp = np.sum(w*x) + b
    if temp <= 0:
        return 0
    else:
        return 1

print("AND(1, 1) >>> ", AND2(1, 1))
# w와 b의 역할
# w는 각 입력 신호가 결과에 주는 영향력(중요도)를 조절하는 매개변수
# b는 뉴런이 얼마나 쉽게 활성화하느냐를 조정하는 매개변수

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-.5, -.5])
    b = .7
    temp = np.sum(w*x) + b
    if temp <= 0:
        return 0
    else:
        return 1

print("NAND(1, 1) >>> ", NAND(1, 1))

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([.5, .5])
    b = -.3
    temp = np.sum(w*x)
    if temp <= 0:
        return 0
    else:
        return 1

print("OR(0, 0) >>> ", OR(0, 0))

# AND, NAND, OR는 모두 같은 구조의 단층 퍼셉트론, 차이는 가중치 매개변수 값

# XOR : 다층 퍼셉트론 (비선형 구조)
def XOR(x1, x2):
    temp = AND2(NAND(x1, x2), OR(x1, x2))
    if temp <= 0:
        return 0
    else:
        return 1

print("XOR >>> ", XOR(1, 0))








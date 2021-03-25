#import sys, os
#sys.path.append(os.pardir) # 부모 디렉토리의 파일을 가져올 수 있도록 설정
from study_deepLearning.dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle
from study_deepLearning.activation_function import sigmoid_function, softmax

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# nomalize : 입력된 이미지의 픽셀값을 0.0 ~ 1.0으로 정규화 (False이면 0~255 사이의 값 유지)
# flatten : 입력 이미지를 1차원 배열로 만들지를 결정
# one_hot_label : 레이블을 원-핫 인코딩 형태로 저장할지를 결정


# 각 데이터의 형상 출력
#print(x_train.shape)
#print(t_train.shape)
#print(x_test.shape)
#print(t_test.shape)

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) # numpy로 저장된 이미지 데이터를 PIL용 데이터 객체로 변
    pil_img.show()

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

# img_show(img)

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid_function(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid_function(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("len(x) >>> " + str(len(x)), "float(accuracy_cnt) >>> " + str(accuracy_cnt))
print("Accuracy: " + str(float(accuracy_cnt)/len(x)))

# 배치처리
batch_size = 100
accuracy_cnt = 0

for i in range(0 , len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy: " + str(float(accuracy_cnt)/len(x)))


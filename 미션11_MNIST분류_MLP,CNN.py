'''
MLP vs CNN
MLP : 영상이나 이미지 처리에 문제발생 -> 픽셀을 변화했을 때 다시 학습해야함
CNN : 특징을 찾아서 학습. Convolution과 Pooling 과정을 반복
'''



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 동일한 실행 결과 확인을 위한 코드입니다.
np.random.seed(123)
tf.random.set_seed(123)

# MNIST 데이터 세트를 불러옵니다.
mnist = tf.keras.datasets.mnist

# MNIST 데이터 세트를 Train set과 Test set으로 나누어 줍니다.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()    

# Train 데이터 5000개와 Test 데이터 500개를 사용합니다.
train_images, train_labels = train_images[:5000].astype(float), train_labels[:5000]
test_images, test_labels = test_images[:500].astype(float), test_labels[:500]

'''
1. 먼저 MLP 모델을 학습해보겠습니다.
'''
print('========== MLP ==========')

# MLP 모델의 입력으로 사용할 수 있도록 (샘플개수, 가로픽셀 * 세로픽셀) 형태로 변환합니다.
train_images = tf.cast(tf.reshape(train_images, (5000, -1)) / 256., tf.float32)
train_labels = tf.convert_to_tensor(train_labels)
test_images = tf.cast(tf.reshape(test_images, (500, -1)) / 256., tf.float32)
test_labels = tf.convert_to_tensor(test_labels)

# MLP 모델을 설정합니다.
MLP_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

# MLP 모델의 학습 방법을 설정합니다.
MLP_model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
              
# 학습을 수행합니다. 
history = MLP_model.fit(train_images, train_labels, epochs = 10, batch_size = 128, verbose = 2)

# MLP 모델 구조를 출력합니다. weight의 수가 52,650개입니다.
MLP_model.summary()

# 평가용 데이터를 활용하여 정확도를 평가합니다.
loss, test_acc = MLP_model.evaluate(test_images, test_labels, verbose = 0)

# 모델 평가 및 예측 결과를 출력합니다.
print('\nMLP Test Loss : {:.4f} | MLP Test Accuracy : {}\n'.format(loss, test_acc))

'''
2. 다음으로, CNN 모델을 학습해보겠습니다.
'''
print('========== CNN ==========')

# CNN 모델의 입력으로 사용할 수 있도록 (샘플개수, 가로픽셀, 세로픽셀, 1) 형태로 변환합니다.
train_images = tf.reshape(train_images, (5000, 28, 28, 1))
test_images = tf.reshape(test_images, (500, 28, 28, 1))

# CNN 모델을 설정합니다.
CNN_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'SAME', input_shape = (28,28,1)),
    tf.keras.layers.MaxPool2D(padding = 'SAME'),
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'SAME'),
    tf.keras.layers.MaxPool2D(padding = 'SAME'),
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'SAME'),
    tf.keras.layers.MaxPool2D(padding = 'SAME'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

# CNN 모델의 학습 방법을 설정합니다.
CNN_model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 학습을 수행합니다. 
history = CNN_model.fit(train_images, train_labels, epochs = 10, batch_size = 128, verbose = 2)

# CNN 모델 구조를 출력합니다. weight의 수가 52,298개입니다.
CNN_model.summary()

# 평가용 데이터를 활용하여 정확도를 평가합니다.
loss, test_acc = CNN_model.evaluate(test_images, test_labels, verbose = 0)

# 모델 평가 및 예측 결과를 출력합니다.
print('\nCNN Test Loss : {:.4f} | CNN Test Accuracy : {}'.format(loss, test_acc))
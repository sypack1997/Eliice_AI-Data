import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

'''
1. 데이터를 생성합니다.
'''
np.random.seed(42)
X = np.random.normal(size = 100)
Y = X * 2 + np.random.normal(size = X.shape)

'''
2. 데이터를 시각화해봅니다.
'''
fig, ax = plt.subplots()
ax.scatter(
    X, 
    Y, 
)
ax.set_xlabel("X")
ax.set_ylabel("Y")

# elice에서 그래프를 확인
fig.savefig("plot.png")


'''
3. 학습용, 테스트용 데이터를 나눕니다. 100개 데이터 중 처음 80개는 학습용, 마지막 20개는 테스트용으로 사용합니다.
'''
train_X = pd.DataFrame(X[:80], columns=['X'])
train_Y = pd.Series(Y[:80])

test_X = pd.DataFrame(X[80:], columns=['X'])
test_Y = pd.Series(Y[80:])

'''
4. 선형 회귀 모델을 생성하고, 학습용 데이터로 학습합니다.
'''
lrmodel = LinearRegression() # 기본 선형회귀 모델
lrmodel.fit(train_X, train_Y)

ridgemodel = Ridge() # Ridge 선형회귀 모델
ridgemodel.fit(train_X, train_Y)

lassomodel = Lasso() # Lasso 선형회귀 모델
lassomodel.fit(train_X, train_Y)

"""
5. test_X에 대해서 예측하고 mean squared error 를 비교합니다.
"""
lr_pred_X = lrmodel.predict(test_X) # predict() 를 이용하여 예측합니다.
lr_mse = mean_squared_error(lr_pred_X, test_Y)
print('LinearRegression MSE : {}'.format(lr_mse))

ridge_pred_X = ridgemodel.predict(test_X) # predict() 를 이용하여 예측합니다.
ridge_mse = mean_squared_error(ridge_pred_X, test_Y)
print('Ridge MSE : {}'.format(ridge_mse))

lasso_pred_X = lassomodel.predict(test_X) # predict() 를 이용하여 예측합니다.
lasso_mse = mean_squared_error(lasso_pred_X, test_Y)
print('Lasso MSE : {}'.format(lasso_mse))

'''
6. 각 모델의 예측 결과를 시각화 해봅니다.
'''
X_arange = np.arange(-3, 3, 0.01).reshape(-1, 1) # -3~3 까지 0.01 간격으로 샘플링

lr_Y = lrmodel.predict(X_arange)
ridge_Y = ridgemodel.predict(X_arange)
lasso_Y = lassomodel.predict(X_arange)

fig, ax = plt.subplots()
ax.scatter(
    X, 
    Y,
    label = 'Data'
)
ax.scatter(
    X_arange, 
    lr_Y,
    label = 'linear regression',
    s = 0.1
)
ax.scatter(
    X_arange, 
    ridge_Y,
    label = 'Ridge',
    s = 0.1
)
ax.scatter(
    X_arange, 
    lasso_Y,
    label = 'Lasso',
    s = 0.1
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()


fig.savefig("plot_result.png")
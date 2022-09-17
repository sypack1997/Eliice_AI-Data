import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# 데이터를 읽고 전 처리합니다
df = pd.read_csv("data/Advertising.csv")
df = df.drop(columns=['Unnamed: 0'])

X = df.drop(columns=['Sales'])
Y = df['Sales']

# K-fold 교차 검증
kf = KFold(n_splits = 5) # K=5 인 K-fold instance 생성

n_try = 1
MSE_arr = []
for train_index, test_index in kf.split(X):
    train_X, test_X = X.loc[train_index], X.loc[test_index]
    train_Y, test_Y = Y.loc[train_index], Y.loc[test_index]

    # 다중 선형 회귀 모델을 초기화 하고 학습합니다
    lrmodel = LinearRegression()
    lrmodel.fit(train_X, train_Y)

    # test_X 의 예측값을 계산합니다
    pred_test = lrmodel.predict(test_X)

    MSE_test = mean_squared_error(test_Y, pred_test) # mean_squared_error() 를 활용해서 MSE를 계산합니다.
    print('MSE_test (%d-fold) : %f' % (n_try, MSE_test))
    
    MSE_arr.append(MSE_test)
    n_try += 1
print('MSE_test (average) : %f'%np.mean(MSE_arr))

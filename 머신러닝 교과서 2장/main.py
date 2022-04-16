import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


class Perceptron(object):
    '''
    (매개변수)
    eta : float
        학습률 (0.0과 1.0사이의 float)
    n_iter : int
        훈련 데이터셋 반복 횟수
    random_state : int
        가중치 무작위 초기화를 위한 난수 생성기 시드

    (속성)
    w_ : 1d-array
        학습된 가중치
    errors_ : list
        에포크마다 누적된 분류 오류
    '''

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        '''
        (매개변수)
        x : n_sample개의 샘플(행)과 n_feature개의 특성(열)으로 이루어진
        훈련 데이터
        y: 타깃값 (1차원 배열)

        (return)
        self: object
        '''
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0:] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        '''입력 계산'''
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        '''단위 계단 함수를 사용하여 클래스 레이블을 반환'''
        return np.where(self.net_input(X) >= 0.0, 1, -1)

def __main__():


    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

    df = pd.read_csv(s, header=None, encoding='utf-8')


    # setosa와 versicolor를 선택합니다
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    # 꽃받침 길이와 꽃잎 길이를 추출합니다
    X = df.iloc[0:100, [0, 2]].values

    # plt.savefig('images/02_06.png', dpi=300)
    plt.show()

    ppn = Perceptron(eta=0.1, n_iter=10)

    ppn.fit(X, y)


__main__()
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



class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        rgen = np.random.RandomState()
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape)
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)

    def net_input(self, X):
        return np.dot(X, self.w_[1:] + self.w_[0])

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


from matplotlib import colors
from numpy.core.fromnumeric import reshape
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    cmap = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, camp=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap[idx], marker=markers[idx],
                    label=cl, edgecolor='black')

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1], facecolors='none', edgecolors='black', alpha=1.0, linewidth=1,
                    marker='o', s=100, label='test_set')
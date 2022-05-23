from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wine = datasets.load_wine()

df = pd.DataFrame(wine['data'], columns=wine['feature_names'])
df['target'] = wine['target']
df.head()

a = 10
b = 6

X = wine.data[:, [a, b]]
y = wine.target

print('클래스 레이블:', np.unique(y))
print('y의 레이블 카운트 : ', np.bincount(y))

# for i in range(df.shape[0]):
#     for j in range(i, ):
#         if i == j or j == 13:
#             continue

#         X = wine.data[:, [i, j]]

#         markers = ('s', 'x', 'o', '^', 'v')
#         cmap=('red', 'blue', 'lightgreen', 'gray', 'cyan')
#         for idx, cl in enumerate(np.unique(y)):
#             plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
#                             alpha=0.8, c=cmap[idx], marker=markers[idx],
#                             label=cl, edgecolor='black')
#         plt.xlabel(wine['feature_names'][i])
#         plt.ylabel(wine['feature_names'][j])
#         plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sc = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

print('y의 레이블 카운트 : ', np.bincount(y))
print('y_train의 레이블 카운트 : ', np.bincount(y_train))
print('y_test의 레이블 카운트 : ', np.bincount(y_test))

sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron


ppn = Perceptron(eta0=0.1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print(f'잘못 분류된 샘플 개수:{(y_test != y_pred).sum()}')

from modules import plot_decision_regions
X_cobined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_cobined_std, y=y_combined, classifier=ppn)
plt.xlabel(wine['feature_names'][a]+'std')
plt.ylabel(wine['feature_names'][b])
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


'''
여기까지 로지스틱 회귀
이상치에 민감해서 할 때마다 다르게 나오는 듯
아래부터는 SVM학습
'''

# from sklearn.svm import SVC
#
# svm=SVC(kernel='linear', C=1.0)
# svm.fit(X_train_std, y_train)
#
# plot_decision_regions(X_cobined_std, y_combined, classifier=svm)
# plt.xlabel(wine['feature_names'][a]+'std')
# plt.ylabel(wine['feature_names'][b])
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

svm = SVC(kernel='rbf', gamma=100, C=10.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_cobined_std, y_combined, classifier=svm)
plt.xlabel(wine['feature_names'][a]+'std')
plt.ylabel(wine['feature_names'][b])
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
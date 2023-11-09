
import numpy as np
import matplotlib.pyplot as plt

def buggy_pca(X, d=1):
    U, S, V = np.linalg.svd(X)
    A = V.T[:, :d]
    projection = X.dot(A).dot(A.T)
    plt.scatter(X[:, 0], X[:, 1], marker='x')
    plt.scatter(projection[:, 0], projection[:, :1], marker='o', color='r')
    plt.title('buggy PCA')
    plt.savefig('buggy_pca.pdf')
    plt.clf()
    # return projection, A, 

def demeaned_pca(X, d=1):
    means = X.mean(axis=0)
    # print(X)
    # print(means)
    # print('means', means.shape)
    X -= means
    # print(X)
    U, S, V = np.linalg.svd(X)
    A = V.T[:, :d]
    projection = X.dot(A).dot(A.T)
    plt.scatter(X[:, 0], X[:, 1], marker='x')
    plt.scatter(projection[:, 0], projection[:, :1], marker='o', color='g')
    # plt.show()
    plt.title('demeaned PCA')
    plt.savefig('demeaned_pca.pdf')
    plt.clf()

def normalized_pca(X, d=1):
    means = X.mean(axis=0)
    std = X.std(axis=0)
    X -= means
    X /= std
    U, S, V = np.linalg.svd(X)
    A = V.T[:, :d]
    projection = X.dot(A).dot(A.T)
    plt.scatter(X[:, 0], X[:, 1], marker='x')
    plt.scatter(projection[:, 0], projection[:, :1], marker='o', color='r')
    # plt.show()
    plt.title('normalized PCA')
    plt.savefig('normalized_pca.pdf')
    plt.clf()

def dro(X, d=1):
    b = np.expand_dims(X.mean(axis=0), 1)
    X_ = X - np.ones((len(X), 1)).dot(b.T)
    U, S, V = np.linalg.svd(X_)
    U = U[:, :d]
    V = V.T[:, :d]
    S = np.diag(S[:d])
    Z = np.sqrt(len(X)) * U
    A = (1/np.sqrt(len(X))) * np.dot(V, S)
    # print(A.shape)
    projection = Z.dot(A.T) + np.ones((len(X), 1)).dot(b.T)
    plt.scatter(X[:, 0], X[:, 1], marker='x')
    plt.scatter(projection[:, 0], projection[:, :1], marker='o')
    # plt.show()
    plt.title('DRO')
    plt.savefig('dro.pdf')
    plt.clf()

data = np.loadtxt('./data/data2D.csv', delimiter=',')
# print('data', data.shape)
buggy_pca(data)
demeaned_pca(data)
normalized_pca(data)
dro(data)

data1000 = np.loadtxt('./data/data1000D.csv', delimiter=',')
U, S, V = np.linalg.svd(data1000)
S = S[:40]
plt.plot([i for i in range(len(S))], S)
plt.xlabel('idx')
plt.ylabel('eigen values')
plt.title('Eigen values for decomposition of the X matrix')
# plt.show()
plt.savefig('d31.pdf')
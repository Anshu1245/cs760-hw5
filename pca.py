
import numpy as np
import matplotlib.pyplot as plt

def buggy_pca(X, d=1):
    U, S, V = np.linalg.svd(X)
    A = V.T[:, :d]
    projection = X.dot(A).dot(A.T) # or "reconstruction"
    plt.scatter(X[:, 0], X[:, 1], marker='x')
    plt.scatter(projection[:, 0], projection[:, :1], marker='o', color='r')
    plt.title('buggy PCA')
    plt.savefig('buggy_pca.pdf')
    plt.clf()
    reconstruction_error = np.linalg.norm(X - projection, 'fro')**2 / len(X)
    print("reconstruction error for buggy pca =", reconstruction_error)
    # return projection, A, 

def demeaned_pca(X, d=1):
    means = X.mean(axis=0)
    X -= means
    U, S, V = np.linalg.svd(X)
    A = V.T[:, :d]
    projection = X.dot(A).dot(A.T)
    plt.scatter(X[:, 0], X[:, 1], marker='x')
    plt.scatter(projection[:, 0], projection[:, :1], marker='o', color='g')
    plt.title('demeaned PCA')
    plt.savefig('demeaned_pca.pdf')
    plt.clf()
    reconstruction_error = np.linalg.norm(X - projection, 'fro')**2 / len(X)
    print("reconstruction error for demeaned pca =", reconstruction_error)
    

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
    plt.title('normalized PCA')
    plt.savefig('normalized_pca.pdf')
    plt.clf()
    reconstruction_error = np.linalg.norm(X - projection, 'fro')**2 / len(X)
    print("reconstruction error for normalized pca =", reconstruction_error)
    

def dro(X, d=1):
    b = np.expand_dims(X.mean(axis=0), 1)
    X_ = X - np.ones((len(X), 1)).dot(b.T)
    U, S, V = np.linalg.svd(X_)
    U = U[:, :d]
    V = V.T[:, :d]
    S = np.diag(S[:d])
    Z = np.sqrt(len(X)) * U
    A = (1/np.sqrt(len(X))) * np.dot(V, S)
    projection = Z.dot(A.T) + np.ones((len(X), 1)).dot(b.T)
    plt.scatter(X[:, 0], X[:, 1], marker='x')
    plt.scatter(projection[:, 0], projection[:, :1], marker='o')
    plt.title('DRO')
    plt.savefig('dro.pdf')
    plt.clf()
    reconstruction_error = np.linalg.norm(X - projection, 'fro')**2 / len(X)
    print("reconstruction error for DRO =", reconstruction_error)
    

data2 = np.loadtxt('./data/data2D.csv', delimiter=',')
data1000 = np.loadtxt('./data/data1000D.csv', delimiter=',')

# eigenvalue plot for 1000D dataset
U, S, V = np.linalg.svd(data1000)
plt.plot([i for i in range(len(S))], S)
plt.xlabel('idx')
plt.ylabel('eigen values')
plt.title('Eigen values for decomposition of the X matrix')
plt.savefig('d.pdf')

# buggy_pca(data1000)
# demeaned_pca(data1000)
# normalized_pca(data1000)
dro(data1000, d=31)
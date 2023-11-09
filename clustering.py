import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.random.seed(1003)

sigma = [0.5, 1, 2, 4, 8]
CENTRES = np.array([[-1, -1], [1, -1], [0, 1]])
obj_kmeans = []
obj_gmm = []
acc_kmeans = []
acc_gmm = []

labels = np.concatenate((np.array([0 for i in range(100)]), np.array([1 for i in range(100)]), np.array([2 for i in range(100)])))

def load_data(sigma=1):
    mean1, cov1 = np.array([-1, -1]), sigma*np.array([[2, 0.5], [0.5, 1]])
    mean2, cov2 = np.array([1, -1]), sigma*np.array([[1, -0.5], [-0.5, 2]])
    mean3, cov3 = np.array([0, 1]), sigma*np.array([[1, 0], [0, 2]])

    data_cluster1 = np.random.multivariate_normal(mean1, cov1, size=100)
    data_cluster2 = np.random.multivariate_normal(mean2, cov2, size=100)
    data_cluster3 = np.random.multivariate_normal(mean3, cov3, size=100)
    data = np.concatenate((data_cluster1, data_cluster2, data_cluster3))
    # data = np.array([data_cluster1, data_cluster2, data_cluster3])
    # print(data.shape)
    # plt.scatter(data_cluster1[:, 0], data_cluster1[:, 1], color='r')
    # plt.scatter(data_cluster2[:, 0], data_cluster2[:, 1], color='b')
    # plt.scatter(data_cluster3[:, 0], data_cluster3[:, 1], color='g')
    # plt.show()

    return data

# data = load_data(0.01)

class GMM():
    def __init__(self, iters=400, num_clusters=3, data=None):
        self.preds = []
        self.iters = iters
        self.clusters = num_clusters
        self.data = data
        if self.data.all()==None:
            raise ValueError('data cannot be None')
        self.w = np.ones(self.clusters)/np.ones(self.clusters).sum()
        self.means = np.random.uniform(low=-5, high=5, size=(self.clusters, self.data.shape[-1]))
        self.covs = np.array([np.identity(self.data.shape[-1]) * np.random.uniform(0.1, high=5) for i in range(self.clusters)]) # + np.array([[0, 1], [1, 0]]) * np.random.uniform(-2, 2)
        # print(self.covs)
        self.soft_labels = np.zeros((self.data.shape[0], self.clusters))
        # print(self.means[0])

    def e_step(self):
        # print('e-step!')
        for n in range(len(self.data)):
            temp_sum = 0
            for c in range(self.clusters):
                dist = multivariate_normal(mean=self.means[c], cov=self.covs[c])
                soft_label_nc = self.w[c] * dist.pdf(self.data[n])
                self.soft_labels[n, c] = soft_label_nc
                temp_sum += soft_label_nc
            self.soft_labels[n, :] /= temp_sum
        
    def m_step(self):
        # print('m-step')
        temp_probs = self.soft_labels.sum(axis=0)
        for c in range(self.clusters):
            temp_mean = np.zeros(self.data.shape[-1])
            temp_cov = np.zeros((self.data.shape[-1], self.data.shape[-1]))
            for n in range(len(self.data)):
                temp_mean += self.soft_labels[n, c] * self.data[n]
                temp_cov += self.soft_labels[n, c] * np.outer((self.data[n] - self.means[c]), (self.data[n] - self.means[c]))
                # print('dot prod\n', self.soft_labels[n, c])
            self.means[c] = temp_mean / temp_probs[c]
            self.covs[c] = temp_cov / temp_probs[c]
        self.w = temp_probs / self.data.shape[0]
        # print('means', self.means)


    def train(self):
        for iter in range(self.iters):
            self.e_step()
            self.m_step()
            # print('iters:', iter, 'nll:', self.nll())

    def nll(self):
        loglikelihood = 0
        for n in self.data:
            temp = 0
            for c in range(self.clusters):
                temp += multivariate_normal.pdf(n, mean=self.means[c], cov=self.covs[c]) * self.w[c]
            loglikelihood += np.log(temp)
        return -loglikelihood
    
    def accuracy(self):
        for idx in range(len(self.data)):
            # print(self.preds)
            self.preds.append(0)
            temp = np.inf
            for c in range(len(CENTRES)):
                dist = np.linalg.norm(self.means[np.argmax(self.soft_labels[idx])]-CENTRES[c])
                if dist < temp:
                    self.preds[idx] = c
                    temp = dist
        self.preds = np.array(self.preds)
        accuracy = (self.preds == labels).sum() / len(labels)
        return accuracy
    
    def objective(self):
        obj = 0
        for idx in range(len(self.data)):
            obj =+ np.linalg.norm(self.data[idx]-self.means[np.argmax(self.soft_labels[idx])])**2
        return obj


    


class KMeans():
    def __init__(self, data, iters=400, k=3):
        self.original_data = data
        self.iters = iters
        self.k = k
        self.load_data(data)
        self.preds = []

    def load_data(self, data):
        self.data = {}
        for n in range(len(data)):
            self.data[n] = [data[n], None]
    
    def train(self):
        idx = np.random.randint(len(self.original_data), size=self.k)
        centres = self.original_data[idx]
        for iters in range(self.iters):
            # print('iter:', iters)
            # assign clusters
            for idx in self.data:
                temp = np.inf
                for c in centres:
                    if ((self.data[idx][0]-c)**2).sum() < temp:
                        temp = ((self.data[idx][0]-c)**2).sum()
                        self.data[idx][1] = c

            # update clusters
            for c in range(self.k):
                temp = []
                for idx in self.data:
                    if (self.data[idx][1]==centres[c]).all():
                        temp.append(self.data[idx][0])
                centres[c] = np.mean(np.array(temp), axis=0)

    def accuracy(self):
        for idx in self.data:
            # print(self.preds)
            self.preds.append(0)
            temp = np.inf
            for c in range(len(CENTRES)):
                dist = np.linalg.norm(self.data[idx][1]-CENTRES[c])
                if dist < temp:
                    self.preds[idx] = c
                    temp = dist
        self.preds = np.array(self.preds)
        accuracy = (self.preds == labels).sum() / len(labels)
        return accuracy
    
    def objective(self):
        obj = 0
        for idx in self.data:
            obj =+ np.linalg.norm(self.data[idx][0]-self.data[idx][1])**2
        return obj





for s in sigma:
    print('sigma =', s)
    data = load_data(s)
    gmm = GMM(data=data)
    gmm.train()
    acc_gmm.append(gmm.accuracy())
    obj_gmm.append(gmm.objective())

    print('gmm acc', acc_gmm[-1])
    print('gmm obj', obj_gmm[-1])
        
                

    km = KMeans(data)
    km.train()
    acc_kmeans.append(km.accuracy())
    obj_kmeans.append(km.objective())

    print('kmeans acc', acc_kmeans[-1])
    print('kmeans obj', obj_kmeans[-1])


plt.plot(sigma, acc_gmm)
plt.plot(sigma, acc_kmeans)
plt.title('Clustering Accuracy')
plt.xlabel('Sigma')
plt.ylabel('Accuracy')
plt.savefig('acc.pdf')
plt.clf()

plt.plot(sigma, obj_gmm)
plt.plot(sigma, obj_kmeans)
plt.title('Clustering Objective')
plt.xlabel('Sigma')
plt.ylabel('Objective')
plt.savefig('obj.pdf')

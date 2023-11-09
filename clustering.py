import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.random.seed(1003)

sigma = [0.5, 1, 2, 4, 8]

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
    def __init__(self, iters=100, num_clusters=3, data=None):
        self.preds = None
        self.iters = iters
        self.clusters = num_clusters
        self.data = data
        if self.data.all()==None:
            raise ValueError('data cannot be None')
        self.w = np.ones(self.clusters)/np.ones(self.clusters).sum()
        self.means = np.random.uniform(low=-5, high=5, size=(self.clusters, self.data.shape[-1]))
        self.covs = np.array([np.identity(self.data.shape[-1]) * np.random.uniform(0.1, high=5) + np.array([[0, 1], [1, 0]]) * np.random.uniform(-2, 2) for i in range(self.clusters)])
        print(self.covs)
        self.soft_labels = np.zeros((self.data.shape[0], self.clusters))
        # print(self.means[0])

    def e_step(self):
        print('e-step!')
        for n in range(len(self.data)):
            temp_sum = 0
            for c in range(self.clusters):
                dist = multivariate_normal(mean=self.means[c], cov=self.covs[c])
                soft_label_nc = self.w[c] * dist.pdf(self.data[n])
                self.soft_labels[n, c] = soft_label_nc
                temp_sum += soft_label_nc
            self.soft_labels[n, :] /= temp_sum
        
    def m_step(self):
        print('m-step')
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
            print('iters:', iter, 'nll:', self.nll())

    def nll(self):
        loglikelihood = 0
        for n in self.data:
            temp = 0
            for c in range(self.clusters):
                temp += multivariate_normal.pdf(n, mean=self.means[c], cov=self.covs[c]) * self.w[c]
            loglikelihood += np.log(temp)
        return -loglikelihood
    
data = load_data(0.1)
# gmm = GMM(iters=300, data=data)
# gmm.train()
# print(gmm.means)
# print(np.argmax(gmm.soft_labels, axis=1))


class KMeans():
    def __init__(self, data, iters=500, k=3):
        self.original_data = data
        self.iters = iters
        self.k = k
        self.load_data(data)

    def load_data(self, data):
        self.data = {}
        for n in range(len(data)):
            self.data[n] = [data[n], None]
    
    def train(self):
        idx = np.random.randint(len(self.original_data), size=self.k)
        centres = self.original_data[idx]
        for iters in range(self.iters):
            print('iter:', iters)
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
        print('centres', centres)





            
            

km = KMeans(data)
km.train()
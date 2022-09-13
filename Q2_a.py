import random as random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
#loading data
iris = datasets.load_iris()
data = iris.data[:, :2] #you can change it to 'data = iris.data' to use all features
label = iris.target #label
# this function use to find datas of each cluster
def find_cluster(data, center, k):
    clusters = {}
    for i in range(k):
        clusters[i] = []
    for data in data:
        euc_dist = []
        for j in range(k):
            euc_dist.append(np.linalg.norm(data - center[j]))
        clusters[euc_dist.index(min(euc_dist))].append(data)
    return clusters
# this function find center of each cluster
def find_center(center, clusters, k):
    new_center =[]
    for i in range(k):
        if len(clusters[i]) == 0:
            new_center.append(np.array(random.choices(data)).flatten())
        else:
            new_center.append(np.nanmean(clusters[i], axis=0))
    return new_center
#this function calculate cost
def costfunc(k, clusters, centers):
    cost = 0
    for i in range(k):
        if len(clusters[i]) !=-0:
            temp = np.array(clusters[i]) - centers[i]
            temp =np.linalg.norm(temp)
            cost = cost + temp
    return cost

#this is main function which use other function for algorithm
def k_means(data,k,iteration):
    clusters = {}
    cost = []
    for i in range(k):
        clusters[i] = []
    centers=random.choices(data, k=k)
    for i in range(iteration):
        clusters = find_cluster(data, centers, k)
        new_centers = find_center(centers, clusters, k)
        new_centers_arr = np.array(new_centers)
        centers_arr = np.array(centers)
        compration = np.equal(new_centers_arr , centers_arr)
        done = compration.all()
        if  done:
            break
        centers = new_centers
        temp_cost = costfunc(k, clusters, centers)
        cost.append(temp_cost)
    return centers , clusters , i , cost
#this function plot clusters and centers
def plot_kmeans(k, clusters, centers):
    plt.figure()
    for i in range(k):
        if  len(clusters[i]) !=0:
            c_temp = np.array(clusters[i])
            plt.scatter(c_temp[:,0], c_temp[:,1], s=50 )
    centroids_arr = np.array(centers)
    plt.scatter(centroids_arr[:, 0], centroids_arr[:, 1], c='black', s=200, alpha=0.5)
#this function calculate score which the question ask for
def evaluation(k, clusters, centers):
    cost = 0
    for i in range(k):
        temp = np.array(clusters[i]) - centers[i]
        temp =np.linalg.norm(temp)
        cost = cost + temp
    diff = 0
    for i in range(k):
        for j in range(k):
            if(i != j):
                tempdiff = np.array(clusters[i]) - centers[j]
                tempdiff =np.linalg.norm(tempdiff)
                diff = diff + tempdiff
    eval = cost / diff
    return eval

#you can change iteration and k
k = 20
iteration = 200
centers , clusters, i ,cost = k_means(data,k,iteration)
plot_kmeans(k, clusters, centers)
iter = range(1,i+1)
plt.figure()
plt.plot(iter, cost)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()
ev = evaluation(k, clusters, centers)
print('If k = ' , k , 'in ' , i , '  iteration evaluation is ' ,ev)
import numpy as np
from numpy import *
import math
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def getXY(dataSet):  # get the X,Y of data
    import numpy as np
    m = shape(dataSet)[0]  # acquire the number of data
    X = []
    Y = []
    for i in range(m):
        X.append(dataSet[i][0])  # get the first dimension
        Y.append(dataSet[i][1])  # get the second dimension
    return np.array(X), np.array(Y)


def showCluster(dataSet, k, clusterAssment, centroids):  # visualize the output of cluster
    fig = plt.figure()
    plt.title("K-means")  # set the title of figure
    ax = fig.add_subplot(111)  # set one subplot
    data = []
    tsne = TSNE()
    dataSet=tsne.fit_transform(dataSet)  # reduce the dimension of data into 2
    centroids=tsne.fit_transform(centroids)  # reduce the dimension of data into 2
    for cent in range(k):  # get the cluster
        ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get the data of a cluster
        data.append(ptsInClust)

    for cent, c, marker in zip(range(k), ['r', 'g'], ['o', '*']):  # draw the scatter plot in red and green, in o and *
        X, Y = getXY(data[cent])
        ax.scatter(X, Y, s=3, c=c, marker=marker)  # draw the dot of sample

    centroidsX, centroidsY = getXY(centroids)
    ax.scatter(centroidsX, centroidsY, s=10, c='black', marker='+', alpha=1)  # draw the dot of center
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.show()

def show_original_img(c_img, c_label):
    fig, ax = plt.subplots(  # build a plot with 10 subplots
        nrows=10,  # rows of plot
        ncols=5,  # columns of plot
        sharex=True,
        sharey=True, )  # share the x and y of subplots
    ax = ax.flatten()  # flatten 2*5 plot into 1*10, which is convenient for use
    for k in range(10):
        for i in range(5):
            rand_num = random.randint(1, 100)  # select number in images of one label randomly
            img = c_img[c_label == k][rand_num]  # select the image that is predicted correctly
            img = img / 255.  # in CIFAR-10, the image is in float32, so have to normalize the images
            ax[i + 5*k].imshow(img, interpolation='nearest')

    ax[0].set_xticks([])  # do not set the ticks on X
    ax[0].set_yticks([])  # do not set the ticks on Y
    y_label = ['airplanes', 'automobiles', 'birds', 'cats', 'deers', 'dogs', 'frogs', 'horses', 'ships' ,'trucks']  # set the label of original images
    for i in range(10):
        ax[i * 5].set_ylabel(y_label[i], rotation = 45, fontsize = 'small')  # set the label of original images
    plt.tight_layout()  # auto adapt the layout of picture
    plt.show()

def show_predicted_img(c_img, seq, p_label):
    fig, ax = plt.subplots(  # build a plot with 10 subplots
        nrows=10,  # rows of plot
        ncols=5,  # columns of plot
        sharex=True,
        sharey=True, )  # share the x and y of subplots
    ax = ax.flatten()  # flatten 2*5 plot into 1*10, which is convenient for use
    for k in range(seq):
        for i in range(5):
            rand_num = random.randint(1, 500)  # select number in images of one label randomly
            img = c_img[p_label == k][rand_num]  # select the image that is predicted correctly
            img = img / 255.  # in CIFAR-10, the image is in float32, so have to normalize the images
            ax[i + 5*k].imshow(img, interpolation='nearest')


    ax[0].set_xticks([])  # do not set the ticks on X
    ax[0].set_yticks([])  # do not set the ticks on Y
    plt.tight_layout()  # auto adapt the layout of picture
    plt.show()

def vectorDistance(v1, v2):  # calculate the Euclidean distance
    return sqrt(sum(power(v1 - v2, 2)))

def compute_Rij(i, j, AssCluster, centers, k):  # calculate Rij
    Mij = vectorDistance(centers[i], centers[j])
    Rij = (sqrt(AssCluster[i, 0]) / AssCluster[i, 1] + sqrt(AssCluster[j, 0]) / AssCluster[j, 1]) / Mij
    return Rij

def compute_Di(i, AssCluster, centers, k):  # calculate the Di
    list_r = []
    for j in range(k):
        if i != j:
            temp = compute_Rij(i, j, AssCluster, centers, k)
            list_r.append(temp)
    return max(list_r)  # get the max of Di

def compute_DB_index(centers, AssCluster, k):
    sigma_R = 0.0
    for i in range(k):
        sigma_R = sigma_R + compute_Di(i, AssCluster, centers, k)  # calculate the sum of Di
    DB_index = float(sigma_R) / float(k)
    return DB_index

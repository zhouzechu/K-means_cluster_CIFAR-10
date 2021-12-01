# -*- coding:utf-8 -*-

import load_data
import utils
from utils import *
import random
import time


def distEclud(vecA, vecB):  # calculate the distance of two vectors
    return sqrt(sum(power(vecA - vecB, 2)))  # using Euclidean distance


def randCenter(dataSet, k):
    return random.sample(list(dataSet), k)  # generate k centers of clusters randomly


def kMeans(dataSet, k, distMeas=distEclud, createCenter=randCenter):
    m = shape(dataSet)[0]  # acquire the number of images
    clusterAssment = mat(zeros((m, 2)))  # create mat to assign image, the first is name of cluster, the second is distance to center
    AssmentCluster = mat(zeros((k, 2)))  # create mat to assign cluster, the first is name of label, the second is total distance in a cluster

    centers = createCenter(dataSet, k)  # generate k centers of clusters randomly
    clusterChanged = True  # judge the cluster of image change or not
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = inf  # set the distance between the image and cluster center to infinity
            minIndex = -1  # set the index of the cluster of the image is -1
            for j in range(k):
                distJI = distMeas(centers[j], dataSet[i])  # calculate the distance between the cluster center and image
                if distJI < minDist:  # if the distance is lower than the distance between the image and cluster center
                    minDist = distJI  # change mindistance
                    minIndex = j  # change index of the cluster of the image
            if clusterAssment[i, 0] != minIndex:  # if the index of the cluster of the image is not suitable
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2  # convenient for reset
        for cent in range(k):  # recalculate center
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centers[cent] = mean(ptsInClust, axis=0)  # calculate mean of cluster
    predict_label = np.empty(m, int)
    for i in range(k):
        for seq in range(m):
            predict_label[seq] = clusterAssment[seq, 0]  # get predicted label of all images
            if clusterAssment[seq, 0] == i:
                AssmentCluster[i, 0] += square(clusterAssment[seq, 1])  # calculate the sum of square of distance
                AssmentCluster[i, 1] = AssmentCluster[i, 1] + 1  # calculate the number of image of a cluster
    return centers, clusterAssment, AssmentCluster, predict_label


if __name__ == '__main__':
    img, c_img, label = load_data.get_imgdata(
        'D:/K-means/cifar-10-python/cifar-10-batches-py/data_batch_3')  # get image and labels
    utils.show_original_img(c_img, label)  # show the original images
    for seq in range(2, 13):
        s_t = time.time()  # set start time
        centers, clusterAss, AssCluster, p_label = kMeans(img, seq)  # k-means clustering
        e_t = time.time()  # set end time
        DBI = compute_DB_index(centers, AssCluster, seq)  # calculate DBI
        print(DBI)
        total = e_t - s_t  # calculate total time
        print(total)
    # utils.showCluster(img, 2, clusterAss, centers)
    # utils.show_original_img(c_img, label)
    # utils.show_predicted_img(c_img, seq, p_label)

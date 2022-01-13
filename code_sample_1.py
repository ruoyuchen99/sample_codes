# ps2_functions.py
# Jay S. Stanley III, Yale University, Fall 2018
# CPSC 453 -- Problem Set 2
#
# This script contains functions for implementing graph clustering and signal processing.
#

import numpy as np
import codecs
import json
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import eigh, sqrtm
from sklearn.cluster import KMeans



def sbm(N, k, pij, pii, sigma):
    """sbm: Construct a stochastic block model

    Args:
        N (integer): Graph size
        k (integer): Number of clusters
        pij (float): Probability of intercluster edges
        pii (float): probability of intracluster edges

    Returns:
        A (numpy.array): Adjacency Matrix
        gt (numpy.array): Ground truth cluster labels
        coords(numpy.array): plotting coordinates for the sbm
    """

    # generate cluster labels

    # generate an array containing the number of points assigned to each cluster
    modulo = N % k
    N_remain = N - modulo
    n_points = int(N_remain/k)
    in_group = np.repeat(n_points, k)
    in_group[:modulo] = n_points+1

    # filling in the ground truth vector with label
    gt = np.zeros(1, dtype = int)
    for i in range(0,k):
        cluster = np.repeat(i,in_group[i])
        gt = np.hstack((gt,cluster))
    gt = gt[1:]

    # construct the affinity matrix

    # generate a symmetric matrix of entries sampled from a uniform distribution for comparison
    uniform = np.random.uniform(size=(N,N))
    uniform = np.triu(uniform)+np.triu(uniform,k=1).T

    # construct a matrix of pii and pij based on cluster label
    probabilities = np.zeros((N,N))
    for i in range(0,k):
        in_cluster=np.where(np.equal(gt,i))[0]
        probabilities[in_cluster, in_cluster[0]:in_cluster[-1]+1] = pii
    probabilities[probabilities==0]=pij

    # comparing the probabilities to the uniform distribution sample to generate affinity
    A = np.less_equal(uniform, probabilities).astype(int)

    # generate coordinates for points
    mean_radian = np.arange(0,2,2/k) * np.pi # generate means evenly spaced around the unit circle
    x = np.cos(mean_radian)[:,np.newaxis]    # converting to Cartesian coordinates
    y = np.sin(mean_radian)[:,np.newaxis]
    mean_cart = np.hstack((x,y))

    # generate full coordinates from multivariate normal distributions
    coords = np.array([[0,0]])
    for i in range(0,k):
        cov = np.diag(np.repeat(sigma,2))
        cluster_size = np.unique(gt[np.where(np.equal(gt,i))], return_counts=True)[1][0]
        coords_cluster = np.random.multivariate_normal(mean_cart[i,:],cov,size=cluster_size)
        coords = np.vstack((coords,coords_cluster))
    coords = coords[1:,:]

    return A, gt, coords


def L(A, normalized=True):
    """L: compute a graph laplacian

    Args:
        A (N x N np.ndarray): Adjacency matrix of graph
        normalized (bool, optional): Normalized or combinatorial Laplacian

    Returns:
        L (N x N np.ndarray): graph Laplacian
    """
    D=np.diag(np.sum(A, axis = 1))
    L = D-A
    if normalized == False:
        return L
    else:
        D_inv_root = sqrtm(np.linalg.inv(D))
        L =  D_inv_root @ L @ D_inv_root
        return L


def compute_fourier_basis(L):
    """compute_fourier_basis: Laplacian Diagonalization

    Args:
        L (N x N np.ndarray): graph Laplacian

    Returns:
        e (N x 1 np.ndarray): graph Laplacian eigenvalues
        psi (N x N np.ndarray): graph Laplacian eigenvectors
    """
    e, psi = eigh(L)
    return e, psi


def gft(s, psi):
    """gft: Graph Fourier Transform (GFT)

    Args:
        s (N x d np.ndarray): Matrix of graph signals.  Each column is a signal.
        psi (N x N np.ndarray): graph Laplacian eigenvectors
    Returns:
        s_hat (N x d np.ndarray): GFT of the data
    """
    s_hat = np.inner(s.T,psi.T).T

    return s_hat


def filterbank_matrix(psi, e, h):
    """filterbank_matrix: build a filter matrix using the input filter h

    Args:
        psi (N x N np.ndarray): graph Laplacian eigenvectors
        e (N x 1 np.ndarray): graph Laplacian eigenvalues
        h (function handle): A function that takes in eigenvalues
        and returns values in the interval (0,1)

    Returns:
        H (N x N np.ndarray): Filter matrix that can be used in the form
        filtered_s = H@s
    """
    h_e = h(e)
    H = psi @ np.diag(h_e) @ psi.T
    return H


def kmeans(X, k, nrep=5, itermax=300):
    """kmeans: cluster data into k partitions

    Args:
        X (n x d np.ndarray): input data, rows = points, cols = dimensions
        k (int): Number of clusters to partition
        nrep (int): Number of repetitions to average for final clustering 
        itermax (int): Number of iterations to perform before terminating
    Returns:
        labels (n x 1 np.ndarray): Cluster labels assigned by kmeans
    """
    n=X.shape[0]

    # initialize array for storing labels for each repetition
    labels_all = np.zeros((n, nrep))
    # initialize array for storing within-cluster distance for each repetition
    d_within_cluster = np.zeros(nrep)

    # perform kmeans
    for rep in range(0,nrep):
        centroids = kmeans_plusplus(X, k)  # find initial centroids
        iter = 0
        labels_old = np.zeros(n)
        while iter<=itermax:

            # Assigning clusters based on the current centroids
            distance_to_centroids = np.zeros((n,k))
            for i in range(0, k):
                centroid = centroids[i, :]
                distance = np.sum(np.power(X - centroid, 2), axis=1) ** 0.5
                distance_to_centroids[:, i] = distance

            # assign new labels by finding the nearest centroids
            labels_new = np.argmin(distance_to_centroids, axis=1)

            # Check if the updated label assignment is sufficiently different
            # from the assignment from the last iteration
            if np.where(np.not_equal(labels_new,labels_old))[0].shape[0]>n/100:
                labels_old=labels_new
            else:
                break

            # Update centroids
            for i in range(0, k):
                cluster = X[np.where(np.equal(labels_new, i))]
                if cluster.shape[0] > 0:
                    centroid = np.mean(cluster, axis=0)
                centroids[i, :] = centroid

            iter+=1
            if iter>itermax:
                print("K-means does not converge within {} iterations".format(itermax))


        # store the label assignment for the current repetition
        labels_all[:,rep] = labels_new

        # calculate and store the total within-cluster distance for this version of label assignment
        distance = 0
        for i in range(0, k):
            cluster = X[np.where(np.equal(labels_new, i))]
            if cluster.shape[0]>0:
                centroid = centroids[i, :]
                distance += np.sum(np.sum(np.power(cluster - centroid, 2), axis=1) ** 0.5)
        d_within_cluster[rep]=distance

    # return the label assignment with minimum within-cluster distance among all repetitions
    labels = labels_all[:,np.argmin(d_within_cluster)].astype(int)

    return labels


def kmeans_plusplus(X, k):
    """kmeans_plusplus: initialization algorithm for kmeans
    Args:
        X (n x d np.ndarray): input data, rows = points, cols = dimensions
        k (int): Number of clusters to partition

    Returns:
        centroids (k x d np.ndarray): centroids for initializing k-means
    """
    n=X.shape[0]

    # choose a single initial point at random to create the first centroid
    centroids = X[np.random.choice(n,1),:]

    #  iteratively find new centroids until we have k ones
    while centroids.shape[0]<k:
        n_c=centroids.shape[0]
        distance_to_centroids = np.zeros((n, n_c))

        # calculate the distance to all centroids for all points
        for i in range(0, n_c):
            centroid = centroids[i, :]
            distance = np.sum(np.power(X - centroid, 2), axis=1) ** 0.5
            distance_to_centroids[:, i] = distance

        # find the nearest centroid for all points
        nearest_centroid = np.argmin(distance_to_centroids, axis=1)

        # find the distance to the nearest centroid for all points
        distance_nc = np.zeros(n)
        for i in range(0, n_c):
            cluster_index = np.where(np.equal(nearest_centroid, i))
            if cluster_index[0].shape[0] > 0:
                distance = distance_to_centroids[cluster_index,i]
                np.put(distance_nc,cluster_index,distance)

        # calculate a probability distribution
        probabilities = distance_nc/np.sum(distance_nc)

        # find a new centroid according to the probability distribution; the point that is the farthest away from its
        # nearest centroid is the most likely to be selected
        new_centroid = X[np.random.choice(n, 1, p=probabilities),:]

        centroids = np.vstack((centroids,new_centroid))

    return centroids


def SC(L, k, psi=None, nrep=5, itermax=300, sklearn=False):
    """SC: Perform spectral clustering 
            via the Ng method
    Args:
        L (np.ndarray): Normalized graph Laplacian
        k (integer): number of clusters to compute
        nrep (int): Number of repetitions to average for final clustering
        itermax (int): Number of iterations to perform before terminating
        sklearn (boolean): Flag to use sklearn kmeans to test your algorithm
    Returns:
        labels (N x 1 np.array): Learned cluster labels
    """
    if psi is None:
        # if no previously calculated bases for the graph Laplacian, compute the first k elements of the Fourier basis
        psi_k = eigh(L)[1][:,:k]
        pass
    else:  # if there is a calculated basis, select the first k eigenvectors
        psi_k = psi[:, :k]

    # normalize eigenvector rows
    l2_norm = np.sum(np.power(psi_k,2), axis=1)**0.5
    psi_norm = (psi_k.T/l2_norm).T

    # perform kmeans using either the sklearn implementation or own implementation
    if sklearn:
        labels = KMeans(n_clusters=k, n_init=nrep,
                        max_iter=itermax).fit_predict(psi_norm)
    else:
        labels = kmeans(psi_norm,k,nrep=nrep,itermax=itermax)

    return labels


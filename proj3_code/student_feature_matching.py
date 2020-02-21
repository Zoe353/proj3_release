import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree,NearestNeighbors
from scipy.spatial.distance import cdist

def compute_feature_distances(features1, features2):
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second set
      features (m not necessarily equal to n)
    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    # n,feat_dim = features1.shape
    # m,feat_dim = features2.shape
    # dists = np.zeros((n,m))
    # for i in range(n):
    #     for j in range(m):
    #         diff = np.linalg.norm(features1[i] - features2[j])
    #         dists[i][j] = diff

    dists = cdist(features1, features2, 'euclidean')

    # raise NotImplementedError('`match_features` function in ' +
    #     '`student_feature_matching.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).
    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).
    You should call `compute_feature_distances()` in this function, and then
    process the output.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second
      set of features (m not necessarily equal to n)
    - x1: A numpy array of shape (n,) containing the x-locations of features1
    - y1: A numpy array of shape (n,) containing the y-locations of features1
    - x2: A numpy array of shape (m,) containing the x-locations of features2
    - y2: A numpy array of shape (m,) containing the y-locations of features2
    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is an
      index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match
    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    dists = compute_feature_distances(features1,features2)
    n,m = dists.shape
    alpha = 0.8
    matches = []
    confidences = []
    for i in range(n):
        indices = np.argpartition(dists[i], 2)[:2]
        dis1 = dists[i][indices[0]]
        dis2 = dists[i][indices[1]]
        candi = indices[0]
        ratio = dis1/dis2
        if ratio <= alpha:
            matches.append([i,candi])
            confidences.append(ratio)
    matches = np.array(matches)
    confidences = np.array(confidences)

    idx = np.argsort(confidences)
    matches = matches[idx]
    confidences = confidences[idx]
    # raise NotImplementedError('`match_features` function in ' +
    #     '`student_feature_matching.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences

def pca(fvs1, fvs2, n_components= 24):
    """
    Perform PCA to reduce the number of dimensions in each feature vector resulting in a speed up.
    You will want to perform PCA on all the data together to obtain the same principle components.
    You will then resplit the data back into image1 and image2 features.

    Helpful functions: np.linalg.svd, np.mean, np.cov

    Args:
    -   fvs1: numpy nd-array of feature vectors with shape (k,128) for number of interest points 
        and feature vector dimension of image1
    -   fvs1: numpy nd-array of feature vectors with shape (m,128) for number of interest points 
        and feature vector dimension of image2
    -   n_components: m desired dimension of feature vector

    Return:
    -   reduced_fvs1: numpy nd-array of feature vectors with shape (k, m) with m being the desired dimension for image1
    -   reduced_fvs2: numpy nd-array of feature vectors with shape (k, m) with m being the desired dimension for image2
    """

    reduced_fvs1, reduced_fvs2 = None, None
    #############################################################################
    # TODO: YOUR PCA CODE HERE                                                  #
    #############################################################################
    k1,d = fvs1.shape
    fvs_stack = np.vstack((fvs1,fvs2))
    fvs_stack = fvs_stack - np.mean(fvs_stack, axis=0)
    U,S,Vh = np.linalg.svd(fvs_stack)
    reduced_fvs =  fvs_stack @ Vh[:n_components].T
    reduced_fvs1 = reduced_fvs[:k1]
    reduced_fvs2 = reduced_fvs[k1:]

    # raise NotImplementedError('`pca` function in ' +
    # '`student_feature_matching.py` needs to be implemented')
    
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return reduced_fvs1, reduced_fvs2

def accelerated_matching(features1, features2, x1, y1, x2, y2):
    """
    This method should operate in the same way as the match_features function you already coded.
    Try to make any improvements to the matching algorithm that would speed it up.
    One suggestion is to use a space partitioning data structure like a kd-tree or some
    third party approximate nearest neighbor package to accelerate matching.
    Note that doing PCA here does not count. This implementation MUST be faster than PCA
    to get credit.
    """

    #############################################################################
    # TODO: YOUR CODE HERE                                                  #
    #############################################################################
    matches = []
    confidences = []
    alpha = 0.5
    k1,d = features1.shape
    features = np.vstack((features1, features2))
    knn = NearestNeighbors(n_neighbors=3,algorithm='kd_tree',leaf_size=100,radius=20)
    knn.fit(features2)
    neigh_dist, neigh_ind = knn.kneighbors(features1, return_distance=True)
    for k in range(k1):
        ratio = neigh_dist[k][0]/neigh_dist[k][1]
        if ratio<=alpha:
            matches.append([k,neigh_ind[k][0]])
            confidences.append(ratio)

    matches = np.array(matches)
    confidences = np.array(confidences)
    idx = np.argsort(confidences)
    matches = matches[idx]
    confidences = confidences[idx]

    # raise NotImplementedError('`accelerated_matching` function in ' +
    # '`student_feature_matching.py` needs to be implemented')
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return matches, confidences





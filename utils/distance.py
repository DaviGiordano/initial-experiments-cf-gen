import numpy as np
import pandas as pd
from sklearn.metrics import DistanceMetric
from sklearn.metrics.pairwise import cosine_distances


def calculate_pairwise_distance(df1, df2, metric):
    """
    Calculate the pairwise distance between two dataframes using the specified distance metric.

    Parameters:
    df1 (pd.DataFrame): The first dataframe.
    df2 (pd.DataFrame): The second dataframe.
    metric (str): The type of distance metric to use. Must be one of 'euclidean', 'manhattan', 'chebyshev', or 'cosine'.

    Returns:
    np.ndarray: A 1D array containing the pairwise distances between corresponding rows of df1 and df2.

    Raises:
    Exception: If an invalid distance metric is provided.
    """
    if metric in ['euclidean', 'manhattan', 'chebyshev']:
        distance_matrix = DistanceMetric\
                        .get_metric(metric)\
                        .pairwise(df1, df2)
    elif metric == 'cosine':
        distance_matrix = cosine_distances(df1, df2)
    else:
        raise Exception('Invalid distance metric')

    return np.diag(distance_matrix)

from scipy.spatial.distance import mahalanobis

def calculate_mahalanobis_distances(vectors, distribution):
    """
    Calculate the Mahalanobis distances between each vector and a distribution.
    
    Parameters:
    vectors (array-like): Array-like structure containing the vectors for which the Mahalanobis distances are to be calculated.
    distribution (array-like): Array-like structure representing the distribution used to calculate the Mahalanobis distances.
    
    Returns:
    np.ndarray: An array of Mahalanobis distances for each vector in vectors.
    """
    
    vectors = np.asarray(vectors)
    distribution = np.asarray(distribution)
    
    cov_matrix = np.cov(distribution.T)
    reg_term = 1e-6 * np.eye(cov_matrix.shape[0])
    cov_matrix += reg_term
    cov_inv = np.linalg.inv(cov_matrix)
    mean_vector = np.mean(distribution, axis=0)
    distances = np.apply_along_axis(lambda row: mahalanobis(row, mean_vector, cov_inv), 1, vectors)

    return distances

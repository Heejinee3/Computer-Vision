import cv2
import numpy as np
from numpy import linalg

from distance import pdist
from feature_extraction import feature_extraction


def get_bags_of_words(image_paths, feature):
    """
    This function assumes that 'vocab_*.npy' exists and contains an N x feature vector
    length matrix 'vocab' where each row is a kmeans centroid or visual word. This
    matrix is saved to disk rather than passed in a parameter to avoid recomputing
    the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path
    :param feature: name of image feature representation.

    :return: an N x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each
        image's histogram ('vocab_size') below.
    """
    vocab = np.load(f'vocab_{feature}.npy')

    vocab_size = vocab.shape[0]

    # Your code here. You should also change the return value.
    num_image = len(image_paths)
    hist = np.zeros((num_image, vocab_size))
    
    for i in range(num_image):
        img = cv2.imread(image_paths[i])[:, :, ::-1]
        features = feature_extraction(img, feature)
        tmp_hist = np.zeros(vocab_size)
        
        dist = pdist(vocab, features)
        min_words = np.argmin(dist, axis=0)
        for min_word in min_words:
            tmp_hist[min_word] = tmp_hist[min_word] + 1
            
        tmp_hist = tmp_hist / np.linalg.norm(tmp_hist)
        hist[i,:] = tmp_hist


    return hist

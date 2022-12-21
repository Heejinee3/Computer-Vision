import cv2
import numpy as np
from numpy import linalg
import math
from distance import pdist
from feature_extraction import feature_extraction


def get_spatial_pyramid_feats(image_paths, max_level, feature):
    """
    This function assumes that 'vocab_*.npy' exists and
    contains an N x feature vector length matrix 'vocab' where each row
    is a kmeans centroid or visual word. This matrix is saved to disk rather than passed
    in a parameter to avoid recomputing the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path,
    :param max_level: level of pyramid,
    :param feature: name of image feature representation.

    :return: an N x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each
        image's histogram ('vocab_size'), multiplies with
        (1 / 3) * (4 ^ (max_level + 1) - 1).
    """

    vocab = np.load(f'vocab_{feature}.npy')

    vocab_size = vocab.shape[0]

    # Your code here. You should also change the return value.
    num_image = len(image_paths)
    feat_num = vocab_size * int((1 / 3) * (4 ** (max_level + 1) - 1))
    hist = np.zeros((num_image, feat_num))
    
    for i in range(num_image):
        img = cv2.imread(image_paths[i])[:, :, ::-1]
        height = img.shape[0]
        width = img.shape[1]
        concat_hist = []
        
        for j in range(max_level + 1):
            two_j = 2**j
            
            for m in range(two_j):
                for n in range(two_j): 
                    min_h = int(height/two_j*m)
                    max_h = int(height/two_j*(m+1))
                    min_w = int(width/two_j*n) 
                    max_w = int(width/two_j*(n+1)) 
                    tmp_img = img[min_h:max_h, min_w:max_w, :]
                    
                    features = feature_extraction(tmp_img, feature)
                    tmp_hist = np.zeros(vocab_size)
                    dist = pdist(vocab, features)
                    min_words = np.argmin(dist, axis=0)
                    for min_word in min_words:
                        tmp_hist[min_word] = tmp_hist[min_word] + 1
                    
                    if j == 0:
                        tmp_hist * (2**(-1*max_level))
                    else:
                        tmp_hist * (2**(-1*max_level + j - 1))
                        
                    concat_hist.append(tmp_hist)
          
                    
        concat_hist = np.array(concat_hist).flatten()           
        concat_hist = concat_hist / np.linalg.norm(concat_hist)
        hist[i,:] = concat_hist

    return hist
    
    
    

    return np.zeros((1500, 36))

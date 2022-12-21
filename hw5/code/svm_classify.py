import numpy as np
from sklearn import svm


def svm_classify(train_image_feats, train_labels, test_image_feats, kernel_type):
    """
    This function should train a linear SVM for every category (i.e., one vs all)
    and then use the learned linear classifiers to predict the category of every
    test image. Every test feature will be evaluated with all 15 SVMs and the
    most confident SVM will 'win'.

    :param train_image_feats:
        an N x d matrix, where d is the dimensionality of the feature representation.
    :param train_labels:
        an N array, where each entry is a string indicating the ground truth category
        for each training image.
    :param test_image_feats:
        an M x d matrix, where d is the dimensionality of the feature representation.
        You can assume M = N unless you've modified the starter code.

    :return:
        an M array, where each entry is a string indicating the predicted
        category for each test image.
    """

    categories = np.unique(train_labels)

    # Your code here. You should also change the return value.
    len_category = len(categories)
    confidence = np.zeros((test_image_feats.shape[0], len_category)) 
    
    for i in range(len_category):
        y_train = (train_labels == categories[i])
        
        if kernel_type == 'RBF':
            model = svm.SVC(kernel='rbf', C = 10, gamma = 0.1)
        if kernel_type == 'linear':
            model = svm.SVC(kernel='linear', C = 10)
        model.fit(train_image_feats, y_train)
        tmp_conf = model.decision_function(test_image_feats)
        
        confidence[:, i] = tmp_conf
        
    max_index = np.argmax(confidence, axis = 1)
        
        
    return categories[max_index]
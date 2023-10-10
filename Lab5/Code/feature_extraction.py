import cv2
import numpy as np


def feature_extraction(img, feature):
    """
    This function computes defined feature (HoG, SIFT) descriptors of the target image.

    :param img: a height x width x channels matrix,
    :param feature: name of image feature representation.

    :return: a N x feature_size matrix.
    """

    if feature == 'HoG':
        # HoG parameters
        win_size = (32, 32)
        block_size = (32, 32)
        block_stride = (16, 16)
        cell_size = (16, 16)
        nbins = 9
        deriv_aperture = 1
        win_sigma = 4
        histogram_norm_type = 0
        l2_hys_threshold = 2.0000000000000001e-01
        gamma_correction = 0
        nlevels = 64

        # Your code here. You should also change the return value.
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins, deriv_aperture, win_sigma, histogram_norm_type, l2_hys_threshold, gamma_correction, nlevels)
        descriptors = hog.compute(img)
        descriptors = np.array(descriptors)
        descriptors = descriptors.reshape(-1,36)

        return descriptors

    elif feature == 'SIFT':

        # Your code here. You should also change the return value.
        grid_size = 20
        sift = cv2.SIFT_create()
        height= img.shape[0]
        width = img.shape[1]
        
        h_g = height // grid_size
        w_g = width // grid_size
        
        kps = []
        for i in range(h_g):
            for j in range(w_g):
                kp_x = grid_size * j + grid_size / 2
                kp_y = grid_size * i + grid_size / 2
                kp = cv2.KeyPoint(kp_x, kp_y, grid_size)
                kps.append(kp)
        
        kps, descriptors = sift.compute(img, kps)
        descriptors = np.array(descriptors)

        return descriptors






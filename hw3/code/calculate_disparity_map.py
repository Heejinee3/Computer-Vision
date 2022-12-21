import numpy as np
import cv2

# HW3-d
# Generate the disparity map from two rectified images.
# Use NCC for the matching cost function.
def calculate_disparity_map(img_left, img_right, window_size, max_disparity):
    # Your code here
    ################################################
    H = img_left.shape[0]
    W = img_left.shape[1]

    pad_num = int((window_size-1)/2)
    img_left = np.pad(img_left,((pad_num, pad_num),(pad_num, pad_num)), 'constant', constant_values = 0)
    img_right = np.pad(img_right,((pad_num, pad_num),(pad_num, pad_num)), 'constant', constant_values = 0)

    cost_vol = np.zeros((H, W, max_disparity))

    for i in range(H):
        for j in range(W):
            window_left = img_left[i:i+window_size, j:j+window_size]
            E_left = np.mean(window_left)
            window_left = window_left - E_left
            abs_left = np.sqrt(np.sum(window_left**2))
            for k in range(max_disparity):
                if j+k <= W-1:
                    window_right = img_right[i:i+window_size, j+k:j+k+window_size]
                    E_right = np.mean(window_right)
                    window_right = window_right - E_right
                    abs_right = np.sqrt(np.sum(window_right**2))
                
                    left_dot_right = np.sum(np.multiply(window_left, window_right))
                    cost_vol[i,j,k] = left_dot_right / (abs_left + 1e-8) / (abs_right + 1e-8)
                else:
                    cost_vol[i,j,k] = -1

    kernel = np.ones((3, 3))/(3**2)
    cost_vol = cv2.filter2D(cost_vol, -1, kernel, borderType=0)
    disparity_map = np.argmax(cost_vol,axis=2)

    ################################################

    return disparity_map

import numpy as np
import cv2

# HW3-c
# Given two homography matrices for two images, generate the rectified image pair.
def rectify_stereo_images(img1, img2, h1, h2):
    # Your code here
    # Hint: Care about alignment of image.
    # In order to superpose two rectified images, you need to create crtain amount of margin.
    # Which means you need to do additional things to get fully warped image (not cropped).
    ################################################
    H1,W1,C1 = img1.shape
    points1 = np.array([[0,0],[0,H1-1],[W1-1,H1-1],[W1-1,0]]).reshape(-1,1,2)
    points1 = points1.astype(np.float32)
    points1 = cv2.perspectiveTransform(points1,h1)

    H2,W2,C2 = img1.shape
    points2 = np.array([[0,0],[0,H2-1],[W2-1,H2-1],[W2-1,0]]).reshape(-1,1,2)
    points2 = points2.astype(np.float32)
    points2 = cv2.perspectiveTransform(points2,h2)

    W_max = np.max([points1[0,0,0],points1[1,0,0],points1[2,0,0],points1[3,0,0],points2[0,0,0],points2[1,0,0],points2[2,0,0],points2[3,0,0]])
    W_min = np.min([points1[0,0,0],points1[1,0,0],points1[2,0,0],points1[3,0,0],points2[0,0,0],points2[1,0,0],points2[2,0,0],points2[3,0,0]])
    H_max = np.max([points1[0,0,1],points1[1,0,1],points1[2,0,1],points1[3,0,1],points2[0,0,1],points2[1,0,1],points2[2,0,1],points2[3,0,1]])
    H_min = np.min([points1[0,0,1],points1[1,0,1],points1[2,0,1],points1[3,0,1],points2[0,0,1],points2[1,0,1],points2[2,0,1],points2[3,0,1]])

    shift_matrix = np.array([[1,0,-1*W_min+10],[0,1,-1*H_min+10],[0,0,1]])
    rectified_W = int(W_max-W_min+20)
    rectified_H = int(H_max-H_min+20)
    img1_rectified = cv2.warpPerspective(img1, h1@shift_matrix, (rectified_W,rectified_H))
    img2_rectified = cv2.warpPerspective(img2, h2@shift_matrix, (rectified_W,rectified_H))

    ################################################

    return img1_rectified, img2_rectified
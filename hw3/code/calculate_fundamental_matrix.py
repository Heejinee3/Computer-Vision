import numpy as np

from normalize_points import normalize_points


# HW3-b
# Implement normalized 8-point algorithm
def calculate_fundamental_matrix(pts1, pts2):
    # Assume input matching feature points have 2D coordinate
    assert pts1.shape[1]==2 and pts2.shape[1]==2
    # Number of matching feature points should be same
    assert pts1.shape[0]==pts2.shape[0]
    # Your code here
    ################################################
    pts1 = pts1[:8,:].T
    pts2 = pts2[:8,:].T
    npts1, T1 = normalize_points(pts1, 2)
    npts2, T2 = normalize_points(pts2, 2)

    A = A = np.zeros((8,9))
    for i in range(8):
        x1 = npts1[0,i]
        x2 = npts2[0,i]
        y1 = npts1[1,i]
        y2 = npts2[1,i]
        
        A[i,0] = x1*x2
        A[i,1] = x1*y2
        A[i,2] = x1
        A[i,3] = y1*x2
        A[i,4] = y1*y2
        A[i,5] = y1
        A[i,6] = x2
        A[i,7] = y2
        A[i,8] = 1

    eig_val, eig_vec= np.linalg.eig(A.T@A)
    f = eig_vec[:,-1]
    F = np.reshape(f,(3,3)).T

    U, pre_S, VT = np.linalg.svd(F)
    S = np.zeros((3,3))
    S[0,0] = pre_S[0]
    S[1,1] = pre_S[1]
    F = U@S@VT
    
    fundamental_matrix = T2.T@F@T1
    
    ################################################

    return fundamental_matrix
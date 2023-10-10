import cv2
import numpy as np


def get_interest_points(image, descriptor_window_image_width):
    
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:,:,0]
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
    H = image.shape[0]
    W = image.shape[1]
    
    Ix = np.zeros((H, W))
    Iy = np.zeros((H, W))
    
    for x in range(W-1):
        Ix[:,x] = image[:, x+1] - image[:, x]
    
    for y in range(H-1):
        Iy[y,:] = image[y+1, :] - image[y, :]
    
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    gIxx = cv2.GaussianBlur(Ixx, (descriptor_window_image_width-1, descriptor_window_image_width-1), 0)
    gIyy = cv2.GaussianBlur(Iyy, (descriptor_window_image_width-1, descriptor_window_image_width-1), 0)
    gIxy = cv2.GaussianBlur(Ixy, (descriptor_window_image_width-1, descriptor_window_image_width-1), 0)
    
    C = gIxx * gIyy - gIxy**2.0 - 0.04 * (gIxx + gIyy)**2.0
    
    x = []
    y = []
    
    for i in range(int(W/descriptor_window_image_width)):
        for j in range(int(H/descriptor_window_image_width)):
            index = np.unravel_index(np.argmax(C[j*descriptor_window_image_width:(j+1)*descriptor_window_image_width, i*descriptor_window_image_width:(i+1)*descriptor_window_image_width], axis=None), (descriptor_window_image_width, descriptor_window_image_width))
            index_x = i*descriptor_window_image_width+index[1]
            index_y = j*descriptor_window_image_width+index[0]
            max_c = C[index_y, index_x]
            if max_c > 0.000005:
                x.append(index_x)
                y.append(index_y)
    
    x = np.array(x)
    y = np.array(y)
    
    return x,y

def get_descriptors(image, x, y, descriptor_window_image_width):
    
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:,:,0]
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    half_w = int(descriptor_window_image_width/2.0)
    quat_w = int(descriptor_window_image_width/4.0)
    image = np.pad(image, ((half_w,half_w),(half_w,half_w)), 'constant', constant_values=0)
    features = np.zeros((x.shape[0], 128))
    
    for i in range(len(x)):
        index_x = int(x[i] + half_w)
        index_y = int(y[i] + half_w)
        f = []
        
        local_img = image[index_y-half_w: index_y+(half_w+1), index_x-half_w: index_x+(half_w+1)]
        Ix = local_img[:descriptor_window_image_width, 1:] - local_img[:descriptor_window_image_width, :descriptor_window_image_width]
        Iy = local_img[1:, :descriptor_window_image_width] - local_img[:descriptor_window_image_width, :descriptor_window_image_width]
        mag_grad = (Ix**2 + Iy**2)**0.5
        theta_grad = np.arctan2(Iy, Ix)
        
        for m in range(4):
            for n in range(4):
                local_theta = theta_grad[quat_w*m:quat_w*(m+1), quat_w*n:quat_w*(n+1)] 
                local_mag = mag_grad[quat_w*m:quat_w*(m+1), quat_w*n:quat_w*(n+1)] 
                
                hist = np.zeros(8)
                for j in range(quat_w):
                    for k in range(quat_w):
                        if (local_theta[j,k] > 0) and (local_theta[j,k] <= np.pi/4):
                            hist[0] = hist[0] + local_mag[j,k]
                        elif (local_theta[j,k] > np.pi/4) and (local_theta[j,k] <= np.pi/2):
                            hist[1] = hist[1] + local_mag[j,k]
                        elif (local_theta[j,k] > np.pi/2) and (local_theta[j,k] <= np.pi/4*3):
                            hist[2] = hist[2] + local_mag[j,k]
                        elif (local_theta[j,k] > np.pi/4*3) and (local_theta[j,k] <= np.pi):
                            hist[3] = hist[3] + local_mag[j,k]
                        elif (local_theta[j,k] > -1*np.pi) and (local_theta[j,k] <= -1*np.pi/4*3):
                            hist[4] = hist[4] + local_mag[j,k]
                        elif (local_theta[j,k] > -1*np.pi/4*3) and (local_theta[j,k] <= -1*np.pi/2):
                            hist[5] = hist[5] + local_mag[j,k]
                        elif (local_theta[j,k] > -1*np.pi/2) and (local_theta[j,k] <= -1*np.pi/4):
                            hist[6] = hist[6] + local_mag[j,k]
                        elif (local_theta[j,k] > -1*np.pi/4) and (local_theta[j,k] <= 0):
                            hist[7] = hist[7] + local_mag[j,k]
                
                f = f + list(hist)
        f = np.array(f)
        f = f/np.linalg.norm(f)
        f[f > 0.2] = 0.2
        f = f/np.linalg.norm(f)
        features[i,:] = f    
    
    return features

def match_features(features1, features2):
    
    matches = []
    confidences = []
    d = {}
 
    for i in range(features1.shape[0]):
        sec_dis = float("inf")
        min_dis = float("inf")

        for j in range(features2.shape[0]):
            distance = np.linalg.norm(features1[i]-features2[j])
            if min_dis >= distance:
                sec_dis = min_dis
                min_dis = distance
                min_j = j
            elif sec_dis >= distance:
                sec_dis = distance
             
        ratio = min_dis/sec_dis
        if ratio <= 0.85:
            d[(i, min_j)] = 1 - ratio
            
    for i in range(features2.shape[0]):
        sec_dis = float("inf")
        min_dis = float("inf")

        for j in range(features1.shape[0]):
            distance = np.linalg.norm(features1[j]-features2[i])
            if min_dis >= distance:
                sec_dis = min_dis
                min_dis = distance
                min_j = j
            elif sec_dis >= distance:
                sec_dis = distance
             
        ratio = min_dis/sec_dis
        if ratio <= 0.85:
            d[(min_j, i)] = 1 - ratio

    for key, value in d.items():
        matches.append(key)
        confidences.append(value)

    matches = np.array(matches)
    confidences = np.array(confidences)
            
    return matches, confidences


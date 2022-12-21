import cv2
import numpy as np


# HW3-a
# Generate rgb image from bayer pattern image
def bayer_to_rgb(bayer_img, mode):
    assert mode=='bilinear' or mode=='bicubic'
    if mode == 'bilinear':
        # Implement demosaicing using bilinear interpolation.
        # Your code here
        ################################################
        W = bayer_img.shape[1]
        H = bayer_img.shape[0]
        
        rgb_img = np.zeros((H, W, 3))
        
        for i in range(H):
            for j in range(W):
                if i % 2 == 0 and j % 2 == 0:
                    rgb_img[i,j,0] = bayer_img[i,j]  
                elif i % 2 == 0 and j % 2 == 1:
                    rgb_img[i,j,1] = bayer_img[i,j] 
                elif i % 2 == 1 and j % 2 == 0:
                    rgb_img[i,j,1] = bayer_img[i,j] 
                else:
                    rgb_img[i,j,2] = bayer_img[i,j]  

        #####################################Fill the blank#####################################
        #Red
        #Horizental blank
        for i in range(0,H-1,2):
            for j in range(1,W-2,2):        
                rgb_img[i,j,0] = round((rgb_img[i,j-1,0]+rgb_img[i,j+1,0])/2) 
        
        #Vertical blank
        for i in range(1,H-2,2):
            for j in range(0,W-1,2):        
                rgb_img[i,j,0] = round((rgb_img[i-1,j,0]+rgb_img[i+1,j,0])/2)

        #Central blank
        for i in range(1,H-2,2):
            for j in range(1,W-2,2):        
                rgb_img[i,j,0] = round((rgb_img[i-1,j-1,0]+rgb_img[i-1,j+1,0]+rgb_img[i+1,j-1,0]+rgb_img[i+1,j+1,0])/4)

        #Green
        #Central blank 1
        for i in range(1,H-2,2):
            for j in range(1,W-2,2):        
                rgb_img[i,j,1] = round((rgb_img[i,j-1,1]+rgb_img[i,j+1,1]+rgb_img[i-1,j,1]+rgb_img[i+1,j,1])/4)
     
        #Central blank 2
        for i in range(2,H-1,2):
            for j in range(2,W-1,2):        
                rgb_img[i,j,1] = round((rgb_img[i,j-1,1]+rgb_img[i,j+1,1]+rgb_img[i-1,j,1]+rgb_img[i+1,j,1])/4)

        #Blue
        #Horizental blank
        for i in range(1,H,2):
            for j in range(2,W-1,2):        
                rgb_img[i,j,2] = round((rgb_img[i,j-1,2]+rgb_img[i,j+1,2])/2)
        
        #Vertical blank
        for i in range(2,H-1,2):
            for j in range(1,W,2):        
                rgb_img[i,j,2] = round((rgb_img[i-1,j,2]+rgb_img[i+1,j,2])/2)
        
        #Central blank
        for i in range(2,H-1,2):
            for j in range(2,W-1,2):        
                rgb_img[i,j,2] = round((rgb_img[i-1,j-1,2]+rgb_img[i-1,j+1,2]+rgb_img[i+1,j-1,2]+rgb_img[i+1,j+1,2])/4)

        #####################################Fill the edge#####################################
        #Red
        #1
        for i in range(H-1):
            rgb_img[i,W-1,0] = rgb_img[i,W-2,0]
        #2
        for j in range(W-1):
            rgb_img[H-1,j,0] = rgb_img[H-2,j,0]
        #3
        rgb_img[H-1,W-1,0] = rgb_img[H-2,W-2,0]

        #Green
        #1
        for j in range(2,W-1,2):
            rgb_img[0,j,1] = round((rgb_img[0,j-1,1]+rgb_img[0,j+1,1]+rgb_img[1,j,1])/3)
        #2
        for j in range(1,W-2,2):
            rgb_img[H-1,j,1] = round((rgb_img[H-1,j-1,1]+rgb_img[H-1,j+1,1]+rgb_img[H-2,j,1])/3)
        #3
        for i in range(2,H-1,2):
            rgb_img[i,0,1] = round((rgb_img[i-1,0,1]+rgb_img[i+1,0,1]+rgb_img[i,1,1])/3)
        #4
        for i in range(1,H-2,2):
            rgb_img[i,W-1,1] = round((rgb_img[i-1,W-1,1]+rgb_img[i+1,W-1,1]+rgb_img[i,W-2,1])/3)
        #5
        rgb_img[H-1,W-1,1] = round((rgb_img[H-1,W-2,1]+rgb_img[H-2,W-1,1])/2)
        #6
        rgb_img[0,0,1] = round((rgb_img[0,1,1]+rgb_img[1,0,1])/2)

        #Blue
        #1
        for i in range(H-1):
            rgb_img[i,0,2] = rgb_img[i,1,2]
        #2
        for j in range(W-1):
            rgb_img[0,j,2] = rgb_img[1,j,2]
        #3
        rgb_img[0,0,2] = rgb_img[1,1,2]

        rgb_img = rgb_img.astype(np.uint8)
        ################################################
    elif mode == 'bicubic':
        # Optional: Implement demosaicing using bicubic interpolation.
        # Your code here
        ################################################
        rgb_img = np.zeros((bayer_img.shape[0], bayer_img.shape[1], 3), dtype=np.uint8)


        ################################################
    
    return rgb_img
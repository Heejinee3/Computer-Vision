import cv2
import numpy as np


def my_filter2D(image, kernel, reflect = False):
    # This function computes convolution given an image and kernel.
    # While "correlation" and "convolution" are both called filtering, here is a difference;
    # 2-D correlation is related to 2-D convolution by a 180 degree rotation of the filter matrix.
    #
    # Your function should meet the requirements laid out on the project webpage.
    #
    # Boundary handling can be tricky as the filter can't be centered on pixels at the image boundary without parts
    # of the filter being out of bounds. If we look at BorderTypes enumeration defined in cv2, we see that there are
    # several options to deal with boundaries such as cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, etc.:
    # https://docs.opencv.org/4.5.0/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5
    #
    # Your my_filter2D() computes convolution with the following behaviors:
    # - to pad the input image with zeros,
    # - and return a filtered image which matches the input image resolution.
    # - A better approach is to mirror or reflect the image content in the padding (borderType=cv2.BORDER_REFLECT_101).
    #
    # You may refer cv2.filter2D() as an exemplar behavior except that it computes "correlation" instead.
    # https://docs.opencv.org/4.5.0/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
    # correlated = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    # correlated = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)   # for extra credit
    # Your final implementation should not contain cv2.filter2D().
    # Keep in mind that my_filter2D() is supposed to compute "convolution", not "correlation".
    #
    # Feel free to add your own parameters to this function to get extra credits written in the webpage:
    # - pad with reflected image content
    # - FFT-based convolution

    ################
    # Your code here
    ################
        
    ker_height = kernel.shape[0]
    ker_width = kernel.shape[1]
    
    if (ker_height % 2 == 0) or (ker_width % 2 == 0):
        print("Filter length is even")
        return
    
    identity_kernel = np.zeros(kernel.shape)
    identity_kernel[int((ker_height - 1) / 2), int((ker_width - 1) / 2)] = 1
    if np.all(kernel == identity_kernel):
        return image
    
    pad_height = int((ker_height - 1) / 2)
    pad_width = int((ker_width - 1) / 2)
    
    if reflect == False:
        if image.ndim == 3:
            padded_image = np.pad(image,((pad_height, pad_height),(pad_width, pad_width),(0, 0)),'constant')
        else:
            padded_image = np.pad(image,((pad_height, pad_height),(pad_width, pad_width)),'constant')
    else:
        if image.ndim == 3:
            padded_image = np.pad(image,((pad_height, pad_height),(pad_width, pad_width),(0, 0)),'reflect')
        else:
            padded_image = np.pad(image,((pad_height, pad_height),(pad_width, pad_width)),'reflect')        

    rotated_kernel = cv2.rotate(kernel,cv2.ROTATE_180)
    result_image = np.zeros(image.shape)

    if image.ndim == 3:
        for i in range(result_image.shape[0]):
            for j in range(result_image.shape[1]):
                result_image[i,j,0] = np.sum(padded_image[i:i+ker_height,j:j+ker_width, 0] * rotated_kernel)
                result_image[i,j,1] = np.sum(padded_image[i:i+ker_height,j:j+ker_width, 1] * rotated_kernel)
                result_image[i,j,2] = np.sum(padded_image[i:i+ker_height,j:j+ker_width, 2] * rotated_kernel)
    else:
        for i in range(result_image.shape[0]):
            for j in range(result_image.shape[1]):
                result_image[i,j] = np.sum(padded_image[i:i+ker_height,j:j+ker_width] * rotated_kernel)
            
    return result_image

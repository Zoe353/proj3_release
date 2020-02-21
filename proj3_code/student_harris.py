import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
import pdb


def get_gaussian_kernel(ksize, sigma):
    """
    Generate a Gaussian kernel to be used in get_interest_points for calculating
    image gradients and a second moment matrix.
    You can call this function to get the 2D gaussian filter.
    
    This might be useful:
    2) Make sure the value sum to 1
    3) Some useful functions: cv2.getGaussianKernel

    Args:
    -   ksize: kernel size
    -   sigma: kernel standard deviation

    Returns:
    -   kernel: numpy nd-array of size [ksize, ksize]
    """
    
    kernel = None
    #############################################################################
    # TODO: YOUR GAUSSIAN KERNEL CODE HERE                                      #
    ker = cv2.getGaussianKernel(ksize,sigma)
    kernel = np.dot(ker,ker.T)
    #############################################################################

    # raise NotImplementedError('`get_gaussian_kernel` function in ' +
    # '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return kernel

def my_filter2D(image, filt, bias = 0):
    """
    Compute a 2D convolution. Pad the border of the image using 0s.
    Any type of automatic convolution is not allowed (i.e. np.convolve, cv2.filter2D, etc.)

    Helpful functions: cv2.copyMakeBorder

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   filt: filter that will be used in the convolution

    Returns:
    -   conv_image: image resulting from the convolution with the filter
    """
    conv_image = None

    #############################################################################
    # TODO: YOUR MY FILTER 2D CODE HERE                                         #
    #############################################################################
    ksize, k = filt.shape
    grid = int(ksize / 2)
    pad_image = np.pad(image,(grid,grid),'constant')
    m,n = pad_image.shape

    conv_image = np.zeros((m,n))

    for i in range(grid,m-grid):
        for j in range(grid,n-grid):
            conv_image[i][j] = np.sum(filt * pad_image[i-grid:i+grid+1,j-grid:j+grid+1])
    conv_image = conv_image[grid:m-grid,grid:n-grid]

    # raise NotImplementedError('`my_filter2D` function in ' +
    # '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return conv_image

def get_gradients(image):
    """
    Compute smoothed gradients Ix & Iy. This will be done using a sobel filter.
    Sobel filters can be used to approximate the image gradient
    
    Helpful functions: my_filter2D from above
    
    Args:
    -   image: A numpy array of shape (m,n) containing the image
               

    Returns:
    -   ix: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the x direction
    -   iy: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the y direction
    """
    
    ix, iy = None, None
    #############################################################################
    # TODO: YOUR IMAGE GRADIENTS CODE HERE                                      #
    #############################################################################
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv_imagex = my_filter2D(image,sobel_x)
    conv_imagey = my_filter2D(image,sobel_y)
    row,col = conv_imagex.shape

    ix = -1*conv_imagex
    iy = conv_imagey
    # raise NotImplementedError('`get_gradients` function in ' +
    # '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return ix, iy


def remove_border_vals(image, x, y, c, window_size = 16):
    """
    Remove interest points that are too close to a border to allow SIFTfeature
    extraction. Make sure you remove all points where a window around
    that point cannot be formed.

    Args:
    -   image: image: A numpy array of shape (m,n,c),
        image may be grayscale of color (your choice)
    -   x: numpy nd-array of shape (m, n)
    -   y: numpy nd-array of shape (m, n)
    -   c: numpy nd-array of shape (m, n)
    -   window_size: int of the window size that we want to remove. (i.e. make sure all
        points in a window_size by window_size area can be formed around a point)
        (set this to 16 for unit testing)

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   c (optional): numpy nd-array of dim (N,) containing the strength
    """

    #############################################################################
    # TODO: YOUR REMOVE BORDER VALS CODE HERE                                   #
    #############################################################################
    # hyperparameter tuning: window-size corresponding to the hyperparameter in the sift alg: feature_length
    window_size = 16
    #================#

    grid = window_size/2
    row, col = image.shape
    n,= x.shape
    new_x = []
    new_y = []
    new_c = []
    for i in range(n):
        if x[i] < grid or x[i] > col-1-grid or y[i] < grid or y[i] > row-1-grid or c[y[i]][x[i]] <=0:
            continue
        else:
            new_x.append(x[i])
            new_y.append(y[i])
            new_c.append(c[y[i]][x[i]])
    x = np.array(new_x)
    y = np.array(new_y)
    c = np.array(new_c)

    # raise NotImplementedError('`remove_border_vals` function in ' +
    # '`student_harris.py` needs to be implemented')
    
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x, y, c

def second_moments(ix, iy, ksize = 7, sigma = 10):
    """
    Given image gradients, ix and iy, compute sx2, sxsy, sy2 using a gaussian filter.

    Helpful functions: my_filter2D

    Args:
    -   ix: numpy nd-array of shape (m,n) containing the gradient of the image with respect to x
    -   iy: numpy nd-array of shape (m,n) containing the gradient of the image with respect to y
    -   ksize: size of gaussian filter (set this to 7 for unit testing)
    -   sigma: deviation of gaussian filter (set this to 10 for unit testing)

    Returns:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: (optional): numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    """

    sx2, sy2, sxsy = None, None, None
    #############################################################################
    # TODO: YOUR SECOND MOMENTS CODE HERE                                       #
    #############################################################################
    sigma  =3
    kernel = get_gaussian_kernel(ksize, sigma)
    sx2 = my_filter2D(ix * ix, kernel)
    sxsy = my_filter2D(ix * iy, kernel)
    sy2 = my_filter2D(iy * iy, kernel)
    # raise NotImplementedError('`second_moments` function in ' +
    # '`student_harris.py` needs to be implemented')
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return sx2, sy2, sxsy

def corner_response(sx2, sy2, sxsy, alpha):

    """
    Given second moments calculate corner resposne.
    R = det(M) - alpha(trace(M)^2)
    where M = [[Sx2, SxSy],
                [SxSy, Sy2]]


    Args:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: (optional): numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    -   alpha: empirical constant in Corner Resposne equaiton (set this to 0.05 for unit testing)

    Returns:
    -   R: Corner response score for each pixel
    """

    R = None
    #############################################################################
    # TODO: YOUR CORNER RESPONSE CODE HERE                                       #
    #############################################################################
    m, n = sx2.shape
    R = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            M = np.array([[sx2[i][j], sxsy[i][j]], [sxsy[i][j], sy2[i][j]]])
            det = np.linalg.det(M)
            trace = np.trace(M)
            R[i][j] = det - alpha * trace * trace

    # raise NotImplementedError('`corner_response` function in ' +
    # '`student_harris.py` needs to be implemented')
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R

def non_max_suppression(R, neighborhood_size = 7):
    """
    Implement non maxima suppression. Take a matrix and return a matrix of the same size
    but only the max values in a neighborhood are non zero. We also do not want local
    maxima that are very small as well so remove all values that are below the median.

    Helpful functions: scipy.ndimage.filters.maximum_filter
    
    Args:
    -   R: numpy nd-array of shape (m, n)
    -   ksize: int that is the size of neighborhood to find local maxima (set this to 7 for unit testing)

    Returns:
    -   R_local_pts: numpy nd-array of shape (m, n) where only local maxima are non-zero 
    """

    R_local_pts = None
    
    #############################################################################
    # TODO: YOUR NON MAX SUPPRESSION CODE HERE                                  #


    #############################################################################

    maxima = maximum_filter(R, size=(neighborhood_size, neighborhood_size))
    diff = R-maxima
    R_local_pts = R
    R_local_pts[diff != 0] = 0
    median = np.median(R_local_pts)
    R_local_pts[R_local_pts < median] = 0

    # raise NotImplementedError('`non_max_suppression` function in ' +
    # '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R_local_pts
    

def get_interest_points(image, n_pts = 1500):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   n_pts: integer of number of interest points to obtain

    Returns:
    -   x: A numpy array of shape (n_pts) containing x-coordinates of interest points
    -   y: A numpy array of shape (n_pts) containing y-coordinates of interest points
    -   R_local_pts: A numpy array of shape (m,n) containing cornerness response scores after
            non-maxima suppression and before removal of border scores
    -   confidences (optional): numpy nd-array of dim (n_pts) containing the strength
            of each interest point
    """
    x, y, R_local_pts, confidences = None, None, None, None
    

    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                               #
    #############################################################################
    ix,iy = get_gradients(image)
    sx2,sy2,sxsy = second_moments(ix,iy)
    R = corner_response(sx2,sy2,sxsy,alpha=0.05)
    R_local_pts = non_max_suppression(R)
    i_candidate = len(R_local_pts[R_local_pts>0])
    if i_candidate < n_pts:
        n_pts = i_candidate
    flat = R_local_pts.flatten()
    indices = np.argpartition(flat, -n_pts)[-n_pts:]
    indices = indices[np.argsort(-flat[indices])]
    indices = np.array(np.unravel_index(indices, R_local_pts.shape))
    y = indices[0]
    x = indices[1]
    confidences = R_local_pts
    x,y,confidences = remove_border_vals(image,x,y,confidences)



    # raise NotImplementedError('`get_interest_points` function in ' +
    # '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return x,y, R_local_pts, confidences



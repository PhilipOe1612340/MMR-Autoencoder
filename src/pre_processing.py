import numpy as np

def add_gaussian_noise(image, mean=0, var=0.01, clip=True):
    """
    Default args taken from OpenCV.
    Args:
        image (ndarray): 
            Input image data. Will be converted to float.
        mean (int, optional): 
            Mean of random distribution. Defaults to 0.
        var (float, optional):
            Variance of random distribution. Defaults to 0.01.
        clip (bool, optional):
            If True (default), the output will be clipped after noise applied. 
            This is needed to maintain the proper image data range. If False, 
            clipping is not applied, and the output may extend beyond the range
            [-1, 1]. Defaults to True.

    Returns:
        noise_image (ndarray)
    """    

    #TODO: Check if need to convert image to float?
    if image.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    noise_image = image + noise
    if clip:
        noise_image = np.clip(noise_image, low_clip, 1.0)
    return noise_image

def getTransformMatrix(src, dst):
    """
    Equation taken from 
    https://math.stackexchange.com/questions/296794/finding-the-transform-matrix-from-4-projected-points-with-javascript

    Args:
        src (ndarray)
        dst (ndarray(channels, rows, cols))
    """
    def computeMatrix(src):
        A = [[src[0][0], src[1][0], src[2][0]],
             [src[0][1], src[1][1], src[2][1]],
             [1, 1, 1]]
        B = [src[3][0], src[3][1], 1]
        
        coe = np.linalg.solve(A, B)
        
        mat = np.array([[src[0][0]*coe[0], src[1][0]*coe[1], src[2][0]*coe[2]],
                        [src[0][1]*coe[0], src[1][1]*coe[1], src[2][1]*coe[2]],
                        coe])
        return mat
    
    src_mat = computeMatrix(src)
    dst_mat = computeMatrix(dst)
    src_mat_inv = np.linalg.inv(src_mat)
    com = dst_mat.dot(src_mat_inv)

    return com

def warpTransform(image, mat):
    """
    Equation taken from OpenCV cv::warpPerspective.
    https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87
    """
    out = np.zeros(image.shape)
    chs, rows, cols = image.shape
    for ch in range(chs):
        for i in range(rows):
            for j in range(cols):
                src_i = (mat[0][0] * i + mat[0][1] * j + mat[0][2]) / \
                        (mat[2][0] * i + mat[2][1] * j + mat[2][2])
                src_j = (mat[1][0] * i + mat[1][1] * j + mat[1][2]) / \
                        (mat[2][0] * i + mat[2][1] * j + mat[2][2])
                #TODO: Interpolation of coordinates
                src_i, src_j = int(src_i), int(src_j)
                out[ch][i][j] = image[ch][src_i][src_j]
    return out

def random_projective_transform(image):
    chs, rows, cols = image.shape
    src = np.float32([[0, 0], [0, 31], [31, 0], [31, 31]])
    dst = np.float32([[1, 1], [10, 28], [20, 0], [15, 30]])
    #TODO: Generate random dest coordinates
    mat = getTransformMatrix(src, dst)
    trans_image = warpTransform(image, mat)
    return trans_image
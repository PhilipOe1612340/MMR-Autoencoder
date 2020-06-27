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
    out = np.zeros(image.shape) # background as black (0, 0, 0)
    chs, rows, cols = image.shape
    for ch in range(chs):
        for i in range(rows):
            for j in range(cols):
                src_i = (mat[0][0] * i + mat[0][1] * j + mat[0][2]) / \
                        (mat[2][0] * i + mat[2][1] * j + mat[2][2])
                src_j = (mat[1][0] * i + mat[1][1] * j + mat[1][2]) / \
                        (mat[2][0] * i + mat[2][1] * j + mat[2][2])
                # without interpolation
                # src_i, src_j = int(src_i), int(src_j)
                # out[ch][i][j] = image[ch][src_i][src_j]

                # bilinear interpolation, taking j (the column index) as x, and i (the row index) as y.
                out[ch][i][j] = computeInterp(image[ch], src_j, src_i)
    return out

def computeInterp(im, x, y):
    """
    compute value on a unexisted float coordinates based on surrounding pixels via bilinear interpolation
    based on https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python

    Args:
        image (ndarray(row, cal)): single channel image
        x (float): x-axis is actually the column index
        y (float): y-axis is the row index
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


def random_projective_transform(image, dst=None):
    """
    Perform random projective transform on image

    Args:
        image (ndarray(ch, row, col))
        dst ((ndarray(4, 2)), optional):
            The destination of points as [[upper left], [upper right], [bottom left], [bottom right]].
            If it is not given as None, a random dst will be computed.
    """
    chs, rows, cols = image.shape
    src = np.float32([[0, 0], [0, cols-1], [rows-1, 0], [rows-1, cols-1]])
    if dst is None:
        #TODO: Generate random dest coordinates
        dst = np.float32([[10, 10], [10, 21], [21, 10], [21, 21]])
    mat = getTransformMatrix(src, dst)
    trans_image = warpTransform(image, mat)
    return trans_image
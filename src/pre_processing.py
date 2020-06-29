import numpy as np
import multiprocessing as mp
import multiprocessing.pool
from functools import partial
import itertools

##TODO: Safe check for images format

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
    np.random.seed()
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

def mp_interp(src, out, mat, cor):
    [ch, i, j] = cor
    src_i = (mat[0][0] * i + mat[0][1] * j + mat[0][2]) / \
            (mat[2][0] * i + mat[2][1] * j + mat[2][2])
    src_j = (mat[1][0] * i + mat[1][1] * j + mat[1][2]) / \
            (mat[2][0] * i + mat[2][1] * j + mat[2][2])
    val = computeInterp(src[ch], src_j, src_i)
    return val

def warpTransform(image, mat):
    """
    Equation taken from OpenCV cv::warpPerspective.
    https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87
    """
    out = np.zeros(image.shape) # background as black (0, 0, 0)
    chs, rows, cols = image.shape
    
    # for ch in range(chs):
    #     for i in range(rows):
    #         for j in range(cols):
    #             src_i = (mat[0][0] * i + mat[0][1] * j + mat[0][2]) / \
    #                     (mat[2][0] * i + mat[2][1] * j + mat[2][2])
    #             src_j = (mat[1][0] * i + mat[1][1] * j + mat[1][2]) / \
    #                     (mat[2][0] * i + mat[2][1] * j + mat[2][2])
    #             # without interpolation
    #             # src_i, src_j = int(src_i), int(src_j)
    #             # out[ch][i][j] = image[ch][src_i][src_j]

    #             # bilinear interpolation, taking j (the column index) as x, and i (the row index) as y.
    #             out[ch][i][j] = computeInterp(image[ch], src_j, src_i)

    # serialize for parallel
    cors = []
    for ch in range(chs):
        for i in range(rows):
            for j in range(cols):
                cors.append([ch, i, j])
    
    # do calculation in parallel
    func = partial(mp_interp, image, out, mat)
    with mp.Pool() as pool:
        vals = pool.map(func, cors)

    # assign value to out
    for index, cor in enumerate(cors):
        [ch, i, j] = cor
        out[ch][i][j] = vals[index]

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

def random_projective_transform(image, dst=None, mirror=False, random_range=0.5):
    """
    Perform random projective transform on image

    Args:
        image (ndarray(ch, row, col))
        dst ((ndarray(4, 2)), optional):
            The destination of points as [[upper left], [upper right], [bottom left], [bottom right]].
            If it is not given as None, a random dst will be computed.
        mirror (boolean, optional)
            allows mirror transformation, which increase the level of distortion
        random_range (float, optional)
            expanded range for random points area, larger range increases the level of distortion.
    """
    chs, rows, cols = image.shape
    src = np.float32([[0, 0], [0, cols-1], [rows-1, 0], [rows-1, cols-1]])
    if dst is None:
        dst = random_dst(rows, cols, mirror, random_range)
    while exist_linear(dst):
        dst = random_dst(rows, cols, mirror, random_range)
    mat = getTransformMatrix(src, dst)
    trans_image = warpTransform(image, mat)
    return trans_image

def random_dst(cols, rows, mirror=False, random_range=0.5):
    """
    Generate random dst points for transformation.
    
    The possible area of each points will be expanded based on the original image size
    and random_range parameter.
    
    For e.g., with random range of 0.25, the dst area (if mirror is False) for upper left
    point [0, 0] in a image of [100, 100] will be ([-25, -25], [-25, 50], [50, -25], [50, 50]),
    which transform to ([0, 0], [0, 75], [75, 0], [75, 75]).

    Args:
        cols (int)
        rows (int)
        mirror (boolean, optional)
            mirror transformation allows four coordinates fall in the area outside of its
            corresponding area
        random_range (float)
            expanded range for random points area
    """
    expanded_i = int(rows * (1 + random_range))
    expanded_j = int(cols * (1 + random_range))
    
    if not mirror:
        # four coordinates will be restricted to their corresponding area
        range_i = expanded_i / 2
        range_j = expanded_j / 2
    else:
        range_i = expanded_i
        range_j = expanded_j

    while True:
        dst = [[0, 0], [0, expanded_j], [expanded_i, 0], [expanded_i, expanded_j]]
        for k in range(4):
            if dst[k][0] == 0:
                np.random.seed()
                dst[k][0] = dst[k][0] + np.random.randint(range_i)
            else:
                np.random.seed()
                dst[k][0] = dst[k][0] - np.random.randint(range_i)
            if dst[k][1] == 0:
                np.random.seed()
                dst[k][1] = dst[k][1] + np.random.randint(range_j)
            else:
                np.random.seed()
                dst[k][1] = dst[k][1] - np.random.randint(range_j)
        if not exist_linear(dst):
            break
    return dst

def exist_linear(p):
    def _exist_linear(p):
        """
        Check linear relationship based on area of triangle
        [ Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By) ] / 2
        """
        if p[0][0] * (p[1][1] - p[2][1]) + p[1][0] * (p[2][1] - p[0][1]) + p[2][0] * (p[0][1] - p[1][1]) == 0:
            return True
        else:
            return False
    
    if _exist_linear([p[0], p[1], p[2]]) or _exist_linear([p[0], p[2], p[3]]) or _exist_linear([p[1], p[2], p[3]]):
        return True
    else:
        return False


"""
Enable non-daemonic process to have children process
"""

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(mp.pool.Pool):
    Process = NoDaemonProcess

def batch_random_projective_transform(images, workers=None, dst=None, mirror=False, random_range=0.5):
    """
    Args:
        images (ndarray(images, channels, rows, cols))
        workers (int, optional)
            If not given, will be cpu count.
        dst (array(4, 2), optional) arg for transform
        mirror (boolean, optional) arg for transform
        random_range (float, optional) arg for transform

    Returns:
        trans_images(ndarray(images, channels, rows, cols))
    """
    altered_transform = partial(random_projective_transform, dst=dst, mirror=mirror, random_range=random_range)
    pool = MyPool(workers)
    trans_images = pool.map(altered_transform, images)
    return trans_images

def batch_gaussian_noise(images, workers=None, mean=0, var=0.01, clip=True):
    """
    Args:
        images (ndarray(images, channels, rows, cols))
        workers ([type], optional)
            If not given, will be cpu count
        mean (int, optional) arg for transform
        var (float, optional) arg for transform
        clip (bool, optional) arg for transform

    Returns:
        noise_images(ndarray(images, channels, rows, cols))
    """
    altered_gaussian = partial(add_gaussian_noise, mean=mean, var=var, clip=clip)
    with mp.Pool(processes=workers) as pool:
        noise_images = pool.map(altered_gaussian, images)
    return noise_images
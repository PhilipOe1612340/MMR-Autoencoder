import numpy as np
import math

# TODO: make a class

def dist(a, b):
    """euclidean distance"""
    return np.linalg.norm(a-b)


def findNClosest(img, imgSet, n=5):
    """Retruns the n closest vectors. The last two rows are the calculated distance and the original index in the array."""
    dimensions = np.shape(imgSet)[1]

    # add column of 0s to the matrix
    zeros = np.zeros((len(imgSet), 2))
    imgSet = np.append(imgSet, zeros, axis=1)

    # put distance value in the last column
    for num, target in enumerate(imgSet, start=0):
        target[dimensions] = dist(img, target[:dimensions])
        target[dimensions + 1] = num

    # sort by last column
    imgSet = imgSet[imgSet[:, dimensions].argsort()]

    # pick n of the first
    return imgSet[:n]



# # Testing with 10000 random vectors
# dim = 4
# exampleImg = np.round(np.random.rand(1, dim) * 100)
# exampleDataSet = np.round(np.random.rand(10000, dim) * 100)
# res = findNClosest(exampleImg, exampleDataSet, 5)
# np.set_printoptions(suppress=True)
# print(" x,      y,     z,       distance,      orig. index")
# print(res)
# print(exampleImg)

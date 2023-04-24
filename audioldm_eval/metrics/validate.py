import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    print("mu1 ", mu1.shape)
    print("mu2 ", mu2.shape)
    print("sigma1 ", sigma1.shape)
    print("sigma2 ", sigma2.shape)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2) * 2.0)

    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


act1 = random(2048 * 2)
act1 = act1.reshape((2, 2048))
act2 = random(2048 * 2)
act2 = act2.reshape((2, 2048))
fid = calculate_fid(act1, act1)
print("FID (same): %.3f" % fid)
fid = calculate_fid(act1, act2)
print("FID (different): %.3f" % fid)

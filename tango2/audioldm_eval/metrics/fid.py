import torch
import numpy as np
import scipy.linalg

# FID评价保真度，越小越好
def calculate_fid(
    featuresdict_1, featuresdict_2, feat_layer_name
):  # using 2048 layer to calculate
    eps = 1e-6
    features_1 = featuresdict_1[feat_layer_name]
    features_2 = featuresdict_2[feat_layer_name]

    assert torch.is_tensor(features_1) and features_1.dim() == 2
    assert torch.is_tensor(features_2) and features_2.dim() == 2

    stat_1 = {
        "mu": np.mean(features_1.numpy(), axis=0),
        "sigma": np.cov(features_1.numpy(), rowvar=False),
    }
    stat_2 = {
        "mu": np.mean(features_2.numpy(), axis=0),
        "sigma": np.cov(features_2.numpy(), rowvar=False),
    }

    # print("Computing Frechet Distance (PANNs)")

    mu1, sigma1 = stat_1["mu"], stat_1["sigma"]
    mu2, sigma2 = stat_2["mu"], stat_2["sigma"]
    assert mu1.shape == mu2.shape and mu1.dtype == mu2.dtype
    assert sigma1.shape == sigma2.shape and sigma1.dtype == sigma2.dtype

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(
            f"WARNING: fid calculation produces singular product; adding {eps} to diagonal of cov"
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            assert False, "Imaginary component {}".format(m)
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return {
        "frechet_distance": float(fid),
    }

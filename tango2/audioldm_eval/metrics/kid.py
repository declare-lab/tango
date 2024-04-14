import torch
import numpy as np
from tqdm import tqdm

# 分多组，每组一定的数量，然后每组分别计算MMD


def calculate_kid(
    featuresdict_1,
    featuresdict_2,
    subsets,
    subset_size,
    degree,
    gamma,
    coef0,
    rng_seed,
    feat_layer_name,
):
    features_1 = featuresdict_1[feat_layer_name]
    features_2 = featuresdict_2[feat_layer_name]

    assert torch.is_tensor(features_1) and features_1.dim() == 2
    assert torch.is_tensor(features_2) and features_2.dim() == 2
    assert features_1.shape[1] == features_2.shape[1]
    if subset_size > len(features_2):
        print(
            f"WARNING: subset size ({subset_size}) is larger than feature length ({len(features_2)}). ",
            "Using",
            len(features_2),
            "for both datasets",
        )
        subset_size = len(features_2)
    if subset_size > len(features_1):
        print(
            f"WARNING: subset size ({subset_size}) is larger than feature length ({len(features_1)}). ",
            "Using",
            len(features_1),
            "for both datasets",
        )
        subset_size = len(features_1)

    features_1 = features_1.cpu().numpy()
    features_2 = features_2.cpu().numpy()

    mmds = np.zeros(subsets)
    rng = np.random.RandomState(rng_seed)

    for i in tqdm(
        range(subsets),
        leave=False,
        unit="subsets",
        desc="Computing Kernel Inception Distance",
    ):
        f1 = features_1[rng.choice(len(features_1), subset_size, replace=False)]
        f2 = features_2[rng.choice(len(features_2), subset_size, replace=False)]
        o = polynomial_mmd(f1, f2, degree, gamma, coef0)
        mmds[i] = o

    return {
        "kernel_inception_distance_mean": float(np.mean(mmds)),
        "kernel_inception_distance_std": float(np.std(mmds)),
    }


def polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1):
    if gamma in [None, "none", "null", "None"]:
        gamma = 1.0 / X.shape[1]
    K = (np.matmul(X, Y.T) * gamma + coef0) ** degree
    return K


def polynomial_mmd(features_1, features_2, degree, gamma, coef0):
    K_XX = polynomial_kernel(
        features_1, features_1, degree=degree, gamma=gamma, coef0=coef0
    )
    K_YY = polynomial_kernel(
        features_2, features_2, degree=degree, gamma=gamma, coef0=coef0
    )
    K_XY = polynomial_kernel(
        features_1, features_2, degree=degree, gamma=gamma, coef0=coef0
    )

    # based on https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)

    diag_X = np.diagonal(K_XX)
    diag_Y = np.diagonal(K_YY)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
    mmd2 -= 2 * K_XY_sum / (m * m)

    return mmd2

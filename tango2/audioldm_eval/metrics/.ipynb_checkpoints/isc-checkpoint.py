import torch
import numpy as np


def calculate_isc(featuresdict, feat_layer_name, rng_seed, samples_shuffle, splits):
    # print("Computing Inception Score")
    
    features = featuresdict[feat_layer_name]

    assert torch.is_tensor(features) and features.dim() == 2
    N, C = features.shape
    if samples_shuffle:
        rng = np.random.RandomState(rng_seed)
        features = features[rng.permutation(N), :]
    features = features.double()

    p = features.softmax(dim=1)
    log_p = features.log_softmax(dim=1)

    scores = []
    for i in range(splits):
        p_chunk = p[(i * N // splits) : ((i + 1) * N // splits), :]  # 一部分的预测概率
        log_p_chunk = log_p[(i * N // splits) : ((i + 1) * N // splits), :]  # log
        q_chunk = p_chunk.mean(dim=0, keepdim=True)  # 概率的均值
        kl = p_chunk * (log_p_chunk - q_chunk.log())  #
        kl = kl.sum(dim=1).mean().exp().item()
        scores.append(kl)
    # print("scores",scores)
    return {
        "inception_score_mean": float(np.mean(scores)),
        "inception_score_std": float(np.std(scores)),
    }

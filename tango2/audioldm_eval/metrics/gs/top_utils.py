import numpy as np


def circle(N=5000):
    phi = 2 * np.pi * np.random.rand(N)
    x = [[np.sin(phi0), np.cos(phi0)] for phi0 in phi]
    x = np.array(x)
    x = x + 0.05 * np.random.randn(N, 2)
    return x


def filled_circle(N=5000):
    ans = []
    while len(ans) < N:
        x = np.random.rand(2) * 2.0 - 1.0
        if np.linalg.norm(x) < 1:
            ans.append(x)
    return np.array(ans) + 0.05 * np.random.randn(N, 2)


def circle_quorter(N=5000):
    phi = np.pi * np.random.rand(N) + np.pi / 2
    x = [[np.sin(phi0), np.cos(phi0)] for phi0 in phi]
    x = np.array(x)
    x = x + 0.05 * np.random.randn(N, 2)
    return x


def circle_thin(N=5000):
    phi = np.random.randn(N)
    x = [[np.sin(phi0), np.cos(phi0)] for phi0 in phi]
    x = np.array(x)
    x = x + 0.05 * np.random.randn(N, 2)
    return x


def planar(N=5000, zdim=32, dim=784):
    A = np.random.rand(N, zdim)
    z = np.random.rand(zdim, dim)
    return np.dot(A, z)

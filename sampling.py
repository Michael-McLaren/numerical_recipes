import numpy as np


def pdf_uniform(x):
    size = x.shape[0]
    g = np.zeros(size)
    g[x <= 0] = 0
    g[(x > 0) & (x < 5)] = 1
    g[x >= 5] = 0
    return g


def pdf(x):
    size = x.shape[0]
    f = np.zeros(size)
    f[x <= 0] = 0
    f[(x > 0) & (x <= 1)] = x[(x > 0) & (x <= 1)] / 2
    f[(x > 1) & (x <= 2)] = 1 - x[(x > 1) & (x <= 2)] / 2
    f[(x > 2) & (x <= 3)] = 0
    f[(x > 3) & (x <= 4)] = (x[(x > 3) & (x <= 4)] - 3) / 2
    f[(x > 4) & (x < 5)] = (5 - x[(x > 4) & (x < 5)]) / 2
    f[x >= 5] = 0
    return f


def generate_triangles(N):
    """Return a numpy array with the length N with
    random numbers following the distribution specified
    """
    vals = np.zeros(N)
    Nvals = 0
    while Nvals < N:
        # generate x
        x = np.random.uniform(-1, 6, size=N)
        f_g = pdf(x) / pdf_uniform(x)
        M = 0.6

        # chance and probability at each x generated
        acc = np.random.uniform(0, 1, N)
        prob = pdf(x) / (M * pdf_uniform(x))

        x_index = np.nonzero(acc < prob)  # index of where acc < prob = True
        new_vals = x[x_index]  # input index into x
        new_vals = new_vals[
            : N - Nvals
        ]  # slice to get the specific amount to make up to N
        vals[Nvals : Nvals + len(new_vals)] = new_vals  # append to vals list
        Nvals = Nvals + len(new_vals)  # update Nvals

    return vals

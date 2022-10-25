import numpy as np
import scipy.optimize
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.stats
import scipy.special as ssp
from typing import Any, Callable
import numpy.typing as npt

D = np.load("data.npz")["D"]
dims = [1, 2, 3, 4, 5]


def blackbox_func(p: npt.NDArray[Any]):
    """This is an example function to optimize


    Args:
        p (npt.NDArray[Any]): 1d array with shape (ndim,)

    Returns:
        __type__: function output
    """
    ndim = len(p)
    pos = dims.index(ndim)
    curD = D[10000 * pos : 10000 * (pos + 1), :ndim]
    return -ssp.logsumexp(np.sum(-0.5 * ((p[None, :] - curD) / 0.3) ** 2, axis=1))


def find_nminima(
    func: Callable[[npt.NDArray[Any]], Any], ndim: int
) -> npt.NDArray[Any]:
    """finds the the minima of a given function with a variable number of dimensions

    Args:
        func (Callable[npt.NDArray[Any]): function
        ndim (int): _description_

    Returns:
        npt.NDArray[Any]: all the unique minima found for the function
    """

    min_list = []
    for i in range(50):  # repeat different inits
        init = (
            np.random.choice([-1, 1]) * 10 * np.random.rand(ndim)
        )  # could just use .uniform but oh well
        min_ = scipy.optimize.minimize(func, x0=init, method="BFGS").x
        min_list.append(min_)
    min_array = np.array(min_list)
    rounded = np.around(min_array, decimals=2)
    num = np.unique(rounded, axis=0)
    return num


def find_nminima_num_loop(func: Callable[[npt.NDArray[Any]], Any]) -> list[int]:
    dims = [1, 2, 3, 4, 5]
    num_list = []
    for ndim in dims:  # loop over each dimension
        num = find_nminima(func, ndim)
        num_list.append(num.shape[0])
    return num_list

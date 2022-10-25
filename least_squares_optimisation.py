import numpy as np
import scipy
import matplotlib as plt

"""
Input data file, find the best period and the best model constants. The period T is between 0.5 and 5.
File is in form (x,y,e) or (inputs, true_output, error)
"""
DATA = np.loadtxt(file)

plt.errorbar(DATA[0], DATA[1], DATA[2], fmt=".")


def modelfunc_(p, x, T):
    """Basic function to optimise over

    Args:
        p (_type_): constants to optimise for
        x (_type_): input data
        T (_type_): the period

    Returns:
        _type_: function output
    """
    model = p[0] * np.sin(2 * np.pi * x / T) + p[1] * np.cos(2 * np.pi * x / T)
    return model


def modelfunc(p, x, y, err, T):
    """function used in the optimisation process.

    Args:
        p (_type_): constants to optimise for
        x (_type_): input data
        y (_type_): true output
        err (_type_): the error
        T (_type_): the period

    Returns:
        _type_: _description_
    """
    model = modelfunc_(p, x, T)
    resid = (y - model) / err
    return resid


def findper():
    """Find the best period and model constants A and B( or p[0] and p[1]).

    Returns:
        _type_: Return tuple of best period and best model constants/parameters/values
    """
    # init values and data
    x, y, e = DATA[0], DATA[1], DATA[2]
    p0 = np.random.rand(2)
    n = 500
    # lists for appending to and looping over
    T_list = np.linspace(0.5, 5, num=n)
    p_list = np.zeros((n, 2))
    res_list = np.zeros(n)
    # loop over periods and  use least squares for A and B
    for i, T in enumerate(T_list):
        result = scipy.optimize.least_squares(modelfunc, p0, args=(x, y, e, T))
        p_list[i] = result.x
        res_list[i] = np.sum(abs(result.fun))

    min_ind = res_list.argmin()  # get min of sum of residues for each model
    model_vals = modelfunc_(p_list[min_ind], DATA[0], T_list[min_ind])  #
    return (T_list[min_ind], model_vals)

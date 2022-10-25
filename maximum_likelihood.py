import numpy as np
from scipy.stats import poisson

"""
Honestly don't remember what this one was about other than maximum likelihood.
"""


def like(p, data_E, data_N):
    E = data_E
    N = data_N
    A1, A2, A3, A4, A5, A6 = p
    eqn = A1 + A2 * np.exp(A3 * E) + A4 * np.exp(-((E - A5) ** 2) / (2 * A6**2))

    like = poisson(eqn).pmf(N)  # each points probability
    loglike = np.sum(np.log(like))  # summing to get loss function to optimise
    return -loglike


def func(p, E):
    A1, A2, A3, A4, A5, A6 = p
    eqn = A1 + A2 * np.exp(A3 * E) + A4 * np.exp(-((E - A5) ** 2) / (2 * A6**2))

    return eqn


def solve_task5():
    """
    Your function needs to return the best period and the uncertainty.
    It also needs to overplot the best model on top of the data
    """
    # YOUR CODE HERE
    p0 = np.random.uniform(1, 10, size=6)
    p0[2] = 1e-10
    p0[4] = DATA["E"].mean()
    res = scipy.optimize.minimize(like, p0, args=(DATA["E"], DATA["N"]))
    p = res.x

    # plotting
    #     print(p)
    #     plt.plot(DATA['E'], DATA['N'], drawstyle='steps')
    #     E = np.linspace(50,79.5)
    #     plt.plot(E, func(p,E))
    #     plt.xlabel('Energy')
    #     plt.ylabel('Number of particles')

    # calculate uncertainties
    inv_HH = res.hess_inv
    error = np.sqrt(np.diag(inv_HH))
    return p[4], error[4]

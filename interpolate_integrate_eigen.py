import numpy as np
import scipy
import scipy.interpolate
import scipy.integrate as integrate
import time


def interpolate():
    """
    This function needs select the best interpolation method for provided data
    and return the numpy array of interpolated values at the locations specified in test.txt
    """
    data = np.loadtxt("data.txt")
    test = np.loadtxt("test.txt")
    test = np.sort(test)
    X, Y = data[0], data[1]

    k = 5  # k cross validations
    error_list = np.zeros(
        (20, k, 22)
    )  # shape is because theres 20 mean errors - 10 times -  and 22 total methods

    x_grid = np.linspace(0, 110, num=500)
    # repeated so it shuffles multiple times and the variance makes less of a difference
    for n in range(20):
        # shuffle the data
        ind = np.arange(X.size)
        ind_shuffle = np.random.permutation(ind)
        X = X[ind_shuffle]
        Y = Y[ind_shuffle]
        split = int(X.size / k)
        # cross validation k = 10
        for i in range(k):
            # split the data
            exclude = ind_shuffle[i * split : (i + 1) * split]
            X_train = x_grid = np.delete(X, exclude)
            Y_train = y_true = np.delete(
                Y, exclude
            )  # exclude the test indices to get the test indices
            x_grid = X[i * split : (i + 1) * split]
            y_true = Y[i * split : (i + 1) * split]
            sort_x = np.argsort(X_train)
            X_train = X_train[sort_x]
            Y_train = Y_train[sort_x]
            # shorten the methods
            uni = scipy.interpolate.UnivariateSpline
            lin = scipy.interpolate.interp1d
            cubic = scipy.interpolate.CubicSpline

            # loops over all the different methods
            for method in range(3):

                if method == 0:
                    j = 0
                    # extra loop for the different smoothing parameters
                    for s1 in np.linspace(
                        0.01, 3, num=20
                    ):  # try all the different s values that could work

                        y_pred = uni(X_train, Y_train, s=s1)(x_grid)
                        error = ((y_true - y_pred) ** 2).mean()
                        error_list[n, i, j] = error
                        j += 1
                elif method == 1:
                    y_pred = cubic(X_train, Y_train)(x_grid)
                    error = ((y_true - y_pred) ** 2).mean()
                    error_list[n, i, 20] = error

                elif method == 2:
                    # clip outside values
                    y_true = y_true[(X_train[0] < x_grid) & (x_grid < X_train[-1])]
                    x_grid = x_grid[(X_train[0] < x_grid) & (x_grid < X_train[-1])]

                    y_pred = lin(X_train, Y_train)(x_grid)
                    error = ((y_true - y_pred) ** 2).mean()
                    error_list[n, i, 21] = error

    # sort the data so its compatible with the interpolate methods
    X, Y = data[0], data[1]
    sort_x = np.argsort(X)
    X, Y = X[sort_x], Y[sort_x]

    # average the means over the methods and find the best one
    choice = error_list.mean(axis=0).mean(axis=0).argmin()
    if choice == 20:
        y_test = scipy.interpolate.CubicSpline(X, Y)(test)
        return y_test
    elif choice == 21:
        y_test = scipy.interpolate.interp1d(X, Y)(test)
        return y_test
    else:
        s2 = np.linspace(0.01, 3, num=20)[choice]
        y_test = scipy.interpolate.UnivariateSpline(X, Y, s=s2)(test)
        return y_test


def hamiltonian_error(N):
    """Finds the percentage difference between the approximated values of the first two eigen values and the true
    values.
    Args:
        N (_type_): The size of the hamiltonian matrix used

    Returns:
        _type_: percentage difference between approx and true values of first two eigen values
    """
    c1 = 0.0380998  # nm^2 eV
    c2 = 1.43996  # nm eV
    r0 = 0.0529177  # nm
    h = 6.62606896e-34  # J s
    c = 299792458.0  # m/s
    hc = 1239.8419  # eV nm
    r_max = 1.5  # nm
    # setting up the diagonals according to the matrix form of the hamiltonian
    v_diag = 2 * c1 * (N / r_max) ** 2 - c2 * (
        np.linspace(1 / N, 1, num=N) * r_max
    ) ** (-1)
    off_diag = np.ones(N - 1) * -c1 * (N / r_max) ** 2

    # calculate first two evals
    evals = scipy.linalg.eigvalsh_tridiagonal(
        v_diag, off_diag, select="i", select_range=[0, 1]
    )

    e_1, e_2 = evals[0], evals[1]
    # true evals
    true_e_1 = -c2 / (2 * r0 * 1**2)
    true_e_2 = -c2 / (2 * r0 * 2**2)
    return abs((e_1 - true_e_1) / true_e_1), abs((e_2 - true_e_2) / true_e_2), e_1, e_2


def potential_numerical(r, alpha):
    """numerically calculates the potential function of an adjusted coulomb law that has additional constant alpha

    Args:
        r (_type_): radius
        alpha (_type_): new constant for coulomb law

    Returns:
        _type_: numerical potential of adjusted coulomb law
    """
    c1 = 0.0380998  # nm^2 eV
    c2 = 1.43996  # nm eV
    r0 = 0.0529177  # nm
    h = 6.62606896e-34  # J s
    c = 299792458.0  # m/s
    hc = 1239.8419  # eV nm
    func = lambda r: (-c2 / r**2) * (r / r0) ** alpha
    V = integrate.quad(func, r, np.inf)
    return V[0]


def calculate_energy_levels_fast():
    """energy level calculation thats quicker using tridiagonal scipy functions

    Returns:
        _type_: first two eigen values of hamiltonian
    """

    c1 = 0.0380998  # nm^2 eV
    c2 = 1.43996  # nm eV
    r0 = 0.0529177  # nm
    h = 6.62606896e-34  # J s
    c = 299792458.0  # m/s
    hc = 1239.8419  # eV nm
    r_max = 1.5  # nm

    N = 100000

    rlist = np.linspace(1 / N, 1, num=N) * r_max

    v_diag = 2 * c1 * (N / r_max) ** 2 - c2 * (
        np.linspace(1 / N, 1, num=N) * r_max
    ) ** (-1)
    off_diag = np.ones(N - 1) * -c1 * (N / r_max) ** 2
    evals = scipy.linalg.eigvalsh_tridiagonal(
        v_diag, off_diag, select="i", select_range=[0, 1]
    )

    return evals[0], evals[1]


def time_it():
    """calculates time and error for calculate_energy_levels_fast. Prints output"""
    c2 = 1.43996  # nm eV
    r0 = 0.0529177  # nm

    t1 = time.time()
    my_e1, my_e2 = calculate_energy_levels_fast()
    t2 = time.time()
    print(f"Calculation took {t2-t1} seconds.")

    e1_th = -c2 / (2 * r0)
    e2_th = e1_th / 4

    er1 = abs((my_e1 - e1_th) / e1_th)
    er2 = abs((my_e2 - e2_th) / e2_th)
    print(f"Err1 = {er1}, Err2 = {er2}.")

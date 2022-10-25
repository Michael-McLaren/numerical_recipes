import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import solve_bvp


"""
System of equations dx/dt, dy/dt, dvx/dt, dvy/dt
Task 1 compute x displacement given conditions. They're written in the code.
Task 2 given x, what velocity and angle would be needed to half the travel time.
"""

m = 0.145  # mass in kg
c = 23.2  # circumference in cm
r = c / 2 / np.pi
A = np.pi * (r) ** 2 / 10000  # m^2
Cd = 0.34

# Earth-related constants
rhoE = 1.19  # air density at sea level kg/m^3
g = 9.80665  # m/s^2


def projectile_motion_air(t, f):
    """
    f0 = x  => dx/dt  = vx
    f1 = y  => dy/dt  = vy
    f2 = vx => dvx/dt = 0
    f3 = vy => dvy/dt = - g
    """
    # Making the assumption that the air is stationary relative to the velocity of the ball
    v2 = f[2] ** 2 + f[3] ** 2
    FD = -(1 / 2) * Cd * A * rhoE * v2
    theta = np.arctan(f[3] / f[2])
    FDx = FD * np.cos(theta)
    FDy = FD * np.sin(theta)

    vals = np.zeros_like(f)
    vals[0] = f[2]
    vals[1] = f[3]
    vals[2] = FDx / m
    vals[3] = -g + FDy / m

    return vals


def height_3_5m(t, f):
    return f[1] - 3.5


def task1():
    # YOUR CODE HERE
    height_3_5m.terminal = True
    height_3_5m.direction = -1

    tspan = [0, 10]

    t_eval = np.linspace(0, 10, num=10000)

    theta = 40 * np.pi / 180
    vx_init = 50 * np.cos(theta)
    vy_init = 50 * np.sin(theta)
    y0 = [0, 1, vx_init, vy_init]
    ans = solve_ivp(
        projectile_motion_air,
        tspan,
        y0,
        t_eval=t_eval,
        events=height_3_5m,
        dense_output=True,
    )
    return ans.y[0, -1]


def derivatives(t, f):
    """
    f0 = x  => dx/dt  = vx
    f1 = y  => dy/dt  = vy
    f2 = vx => dvx/dt = FDx
    f3 = vy => dvy/dt = - g + FDy
    """
    v2 = f[2] ** 2 + f[3] ** 2
    FD = -(1 / 2) * Cd * A * rhoE * v2
    theta = np.arctan(f[3] / f[2])
    FDx = FD * np.cos(theta)
    FDy = FD * np.sin(theta)

    eq0 = f[2]
    eq1 = f[3]
    eq2 = FDx / m
    eq3 = -g + FDy / m
    return np.vstack((eq0, eq1, eq2, eq3))


def bc(fa, fb):
    """
    fa is [x, y, vx, vy] on the left (t = 0)
    fb is [x, y, vx, vy] on the right (t = 5.2086/2)

    init conditions as x = 0, y = 1 at t = 0
    final conditions as x = 125.938, y = 3.5 at t = 5.2086/2
    """
    return np.array([fa[0], fa[1] - 1, fb[0] - 125.938, fb[1] - 3.5])


def task2():
    # YOUR CODE HERE
    t = np.linspace(0, 5.2086 / 2, 100)
    f = 20 * np.random.random((4, t.size))
    sol = solve_bvp(derivatives, bc, t, f, verbose=2)
    t_sol = np.linspace(0, 5.2086 / 2, 1000)
    f_sol = sol.sol(t_sol)
    vx = sol.y[2, 0]
    vy = sol.y[3, 0]
    v = (vx**2 + vy**2) ** (1 / 2)
    theta = np.arctan(vy / vx)
    return v, theta * 180 / np.pi


"""
Make an animation of these things
"""


def proj_5():
    # YOUR CODE HERE
    tspan = [0, 10]

    t_eval = np.linspace(0, 10, num=1000)

    theta = 40 * np.pi / 180
    vx_init = 50 * np.cos(theta)
    vy_init = 50 * np.sin(theta)
    y0 = [0, 1, vx_init, vy_init]
    ans = solve_ivp(
        projectile_motion_air,
        tspan,
        y0,
        t_eval=t_eval,
        events=height_3_5m,
        dense_output=True,
    )

    len_time = len(
        ans.t[ans.t < ans.t_events[0]]
    )  # number of timesteps under before the event is satisfied
    x1, y1 = ans.y[0], ans.y[1]
    return x1[:len_time], y1[:len_time]


def proj_2(v_init, theta_init):
    # YOUR CODE HERE
    tspan = [0, 10]

    t_eval = np.linspace(0, 10, num=2000)

    theta = theta_init * np.pi / 180
    vx_init = v_init * np.cos(theta)
    vy_init = v_init * np.sin(theta)
    y0 = [0, 1, vx_init, vy_init]
    ans = solve_ivp(
        projectile_motion_air,
        tspan,
        y0,
        t_eval=t_eval,
        events=height_3_5m,
        dense_output=True,
    )

    len_time = len(
        ans.t[ans.t < ans.t_events[0]]
    )  # number of timesteps under before the event is satisfied
    x2, y2 = ans.y[0], ans.y[1]
    return x2[:len_time], y2[:len_time]


# YOUR CODE HERE
fig, ax = plt.subplots()
ax.set_xlim(0, 126)
ax.set_ylim(0, 50)
x1, y1 = proj_5()
x2, y2 = proj_2(v, theta)
l1 = ax.plot(x1, y1, "--")
l2 = ax.plot(x2, y2, "g--")

print(x1.shape, x2.shape)


(ball,) = ax.plot([x1[0]], [y1[0]], "ro")
(ball2,) = ax.plot([x2[0]], [y2[0]], "ro")


def animate(i):
    ball.set_data(x1[i], y1[i])
    ball2.set_data(x2[i], y2[i])
    return (
        ball,
        ball2,
    )

import numpy as np


def dw_deps(x0, delta, patch, dist, tau, Y, kernel):
    if kernel == "exponential":
        phi_p = lambda x : -np.exp(-x);
    else:
        assert 1==0, "Kernel %s is not (yet) available in SURE based decision..." %kernel
    x1 = [x0[0] + delta[0], x0[1]+delta[1]]
    distance = dist[x0[0], x0[1], delta[0], delta[1]]
    return 1/(tau**2) * phi_p(distance/(2*tau**2)) * ( patch[0, 0] * (Y[x0[0], x0[1]] - Y[x1[0], x1[1]]) +  patch[x0[0]-x1[0], x0[1]-x1[1]] * (Y[x0[0], x0[1]] - Y[2*x0[0] - x1[0], 2*x0[1]-x1[1]]) )


def df_deps(x, patch, dist, tau, Y, kernel, locality_constraint):
    if kernel == "exponential":
        phi = lambda x : np.exp(-x);
        weights = exponential_kernel(dist[x[0], x[1], :, :], tau, normalized = False)
    else:
        assert 1==0, "Kernel %s is not (yet) available in SURE based decision..." %kernel

    Cx = np.sum(weights)
    term1 = phi(0) / Cx
    term2 = 0
    fact3_0 = 0
    fact3_1 = 0
    for dx in range(-locality_constraint, locality_constraint+1):
        for dy in range(-locality_constraint, locality_constraint+1):
            x2 = [min(max(x[0] + dx, 0), Y.shape[0]),  min(max(x[1], dy), Y.shape[1])]

            term2 += 1/Cx * Y(x2[0], x2[1]) * dw_deps(x0, [dx, dy], patch, dist, tau, Y, kernel)
            fact3_0 += 1/Cx * Y(x2[0], x2[1]) * weights[x2[0], x2[1]]
            fact3_1 += 1/Cx * dw_deps(x0, [dx, dy], patch, dist, tau, Y, kernel)

    term3 = fact3_0 * fact3_1

    return term1 + term2 - term3


def risk(x, patch, dist, tau, Y, f_bar, kernel, locality_constraint, sigma):
    term0 = (Y[x[0], x[1]] - f_bar[x[0], x[1]])**2
    term1 = 2*sigma**2 * df_deps(x, patch, dist, tau, Y, kernel, locality_constraint)

    return term1 + term2 - sigma**2

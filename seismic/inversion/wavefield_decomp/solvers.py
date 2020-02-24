#!/usr/bin/env python
# coding: utf-8
"""
Objective function minimization solvers.
"""

import numpy as np
from collections import deque

from scipy.optimize import Bounds
# from scipy.optimize import OptimizeResult
from sortedcontainers import SortedList

# DEBUG ONLY
import matplotlib.pyplot as plt


# pylint: disable=invalid-name

class SolverGlobalMhMcmc:
    """
    Drop-in custom solver for scipy.optimize.minimize, based on Metrolpolis-Hastings Monte Carlo
    Markov Chain random walk with burn-in and adaptive acceptance rate, followed by N-dimensional
    clustering.

    Rather than returning one global solution, the solver returns the N best ranked local solutions.
    It also returns a probability distribution of each unknown based on Monte Carlo statistics.

    """
    pass
# end class


class HistogramIncremental():
    """Class to incrementally accumulate N-dimensional histogram stats at runtime.
    """
    def __init__(self, bounds, nbins=20):
        self.ndims = len(bounds.lb)
        assert len(bounds.ub) == self.ndims
        self.bins = np.linspace(bounds.lb, bounds.ub, nbins + 1).T
        self.hist = np.zeros((self.ndims, nbins), dtype=int)
    # end func

    def __iadd__(self, x):
        assert len(x) == self.ndims
        for i, _x in enumerate(x):
            idx = np.digitize(_x, self.bins[i, :])
            self.hist[i, idx - 1] += 1
        # end for
        return self
    # end func

# end class


def _propose_step(x, bounds):
    ndims = len(x)
    while True:
        dim = np.random.randint(0, ndims)
        x_new = x[dim] + _propose_step.stepsize[dim]*np.random.randn()
        if x_new >= bounds.lb[dim] and x_new <= bounds.ub[dim]:
            break
        # end if
    # end while
    result = x.copy()
    result[dim] = x_new
    return result
# end func


# Compose as global function first, then abtract out into classes.
def optimize_minimize_mhmcmc_cluster(objective, bounds, args=(), x0=None, T=1, N=3, burnin=1000000, maxiter=10000000,
                                     target_ar=0.5):
    """
    Minimize objective function and return up to N solutions.

    :param objective: Objective function to minimize
    :param bounds: Bounds of the parameter space.
    :param args: Any additional fixed parameters needed to completely specify the objective function.
    :param x0: Initial guess. If None, will be selected randomly and uniformly within the parameter bounds.
    :param T: The "temperature" parameter for the accept or reject criterion.
    :param N: Maximum number of minima to return
    :param burnin: Number of random steps to discard before starting to accumulate statistics.
    :param maxiter: Maximum number of steps to take (including burnin).
    :param target_ar: Target acceptance rate of point samples generated by stepping.
    :return: OptimizeResult containing solution(s) and solver data.
    """
    assert maxiter > burnin, "maxiter {} not greater than burnin steps {}".format(maxiter, burnin)
    main_iter = maxiter - burnin

    beta = 1.0/T

    # DEBUG ONLY:
    np.random.seed(20200220)

    if x0 is None:
        x0 = np.random.uniform(bounds.lb, bounds.ub)
    # end if
    assert np.all((x0 >= bounds.lb) & (x0 <= bounds.ub))
    x = x0.copy()
    funval = objective(x, *args)

    sigma0 = 0.15*(bounds.ub - bounds.lb)
    _propose_step.stepsize = sigma0.copy()

    x_queue = deque(maxlen=10000)
    x_queue.append(x)
    rejected_randomly = 0
    accepted_burnin = 0
    for _ in range(burnin):
        x_new = _propose_step(x, bounds)
        funval_new = objective(x_new, *args)
        log_alpha = -(funval_new - funval)*beta
        if log_alpha > 0 or np.log(np.random.rand()) <= log_alpha:
            x = x_new
            funval = funval_new
            x_queue.append(x)
            accepted_burnin += 1
        elif log_alpha <= 0:
            rejected_randomly += 1
        # end if
    # end for
    ar = float(accepted_burnin)/burnin
    print("Burnin acceptance rate: {}".format(ar))

    pts = np.array(x_queue)
    plt.scatter(pts[:, 0], pts[:, 1], alpha=0.1, s=5)
    plt.axis('equal')
    plt.xlim(bounds.lb[0], bounds.ub[0])
    plt.ylim(bounds.lb[1], bounds.ub[1])
    plt.show()

    # print("Mean: {}".format(np.mean(pts.T, axis=-1)))
    # print("Cov: {}".format(np.cov(pts.T)))

    del x_queue

    accepted = 0
    rejected_randomly = 0

    minima = SortedList(key=lambda rec: rec[1])
    hist = HistogramIncremental(bounds, nbins=50)
    # Cached a lot of potential minimum values, as these need to be clustered before return N results
    N_cached = 3*N
    for _ in range(main_iter):
        x_new = _propose_step(x, bounds)
        funval_new = objective(x_new, *args)
        log_alpha = -(funval_new - funval)*beta
        if log_alpha > 0 or np.log(np.random.rand()) <= log_alpha:
            x = x_new
            funval = funval_new
            minima.add((x, funval))
            if len(minima) > N_cached:
                minima.pop()
            # end if
            hist += x
            accepted += 1
        elif log_alpha <= 0:
            rejected_randomly += 1
        # end if
    # end for
    ar = float(accepted)/main_iter
    print("Acceptance rate: {}".format(ar))
    print("Best minima: {}".format(np.array([_mx[0] for _mx in minima[0:10]])))

    plt.figure(figsize=(12, 20))
    for i in range(hist.ndims):
        plt.subplot(2, 1, i + 1)
        plt.bar(hist.bins[i, :-1] + 0.5*np.diff(hist.bins[i, :]), hist.hist[i, :])
        # plt.xticks(hist.bins[i, :])
        plt.xlabel('x{}'.format(i))
        plt.ylabel('Counts')
    # end for
    plt.show()

# end func


def _obj_fun(x, mu, cov):
    # The number returned from the function must be non-negative.
    # The exponential of the negative of this value is the probablity.
    # x2fac = np.sqrt(np.matmul(np.matmul((x - mu).T, cov), x - mu))
    x2fac = np.sqrt(np.matmul(np.matmul(x - mu, cov), (x - mu).T))
    return x2fac
# end func


def main():
    # DEV TESTING

    # Test functions as per https://en.wikipedia.org/wiki/Test_functions_for_optimization
    from landscapes.single_objective import sphere, himmelblau, easom, rosenbrock

    bounds = Bounds(np.array([-3, -3]), np.array([3, 3]))
    optimize_minimize_mhmcmc_cluster(sphere, bounds, burnin=10000, maxiter=50000)

    bounds = Bounds(np.array([-5, -5]), np.array([5, 5]))
    optimize_minimize_mhmcmc_cluster(lambda xy: 0.1*himmelblau(xy), bounds, burnin=10000, maxiter=50000)

    bounds = Bounds(np.pi + np.array([-3, -2]), np.pi + np.array([3, 4]))
    optimize_minimize_mhmcmc_cluster(lambda xy: 5*(1 + easom(xy)), bounds, burnin=10000, maxiter=50000)

    bounds = Bounds(np.array([-4, -4]), np.array([4, 4]))
    optimize_minimize_mhmcmc_cluster(lambda xy: 0.001*rosenbrock(xy), bounds, burnin=10000, maxiter=50000)

    # Custom test function
    mu = np.array([0, 1])
    cov = np.array([[5, -6.0], [-6.0, 20.0]])
    fixed_args = (mu, cov)
    bounds = Bounds(np.array([-3, -2]), np.array([3, 4]))
    optimize_minimize_mhmcmc_cluster(_obj_fun, bounds, fixed_args, burnin=10000, maxiter=50000)
# end func


if __name__ == "__main__":
    main()
# end if

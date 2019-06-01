"""
This module defines functions and classes for commonly used
statistical operations.

"""

# Many of the following functions have been adapted from the Gnu
# Scientific Library.

import random as _rnd
import math as _math


def rng_uniform(a=0, b=1):
    """
    Generate random numbers from a uniform distribution

    Parameters
    ----------
    a: float
        The lower bound of the interval [a, b]
    b: float
        The upper bound of the interval [a, b]

    Returns
    -------
    float
        Random uniformly distributed numbers in the range [a, b]

    """
    return _rnd.uniform(a, b)


def rng_normal(sigma=1, trunc=None):
    """
    Generate random numbers from a normal distribution

    Parameters
    ----------
    sigma: float
        The standard deviation of the distribution
    trunc: None, tuple
        Specifies whether the distribution is truncated. If it's
        not None then it must be a 2-tuple indicating the range where
        the distribution is defined.

    Returns
    -------
    float
        Random normally distributed numbers, optionally in a
        specific range if the distribution is truncated.

    """
    while True:
        v = _rnd.gauss(0, sigma)
        if not trunc or trunc[0] <= v <= trunc[1]:
            break
    return v


def rng_cauchy_lorentz(gamma=1, trunc=None):
    """
    Generate random numbers from a Cauchy-Lorentz distribution

    Parameters
    ----------
    gamma: float
        The scale of the distribution
    trunc: None, tuple
        Specifies whether the distribution is truncated. If it's
        not None then it must be a 2-tuple indicating the range where
        the distribution is defined.

    Returns
    -------
    float
        Random Cauchy-Lorentz distributed numbers, optionally in a
        specific range if the distribution is truncated.

    """
    while True:
        while True:
            u = _rnd.random()
            if u != 0.5:
                break
        v = gamma * _math.tan(_math.pi * u)
        if not trunc or trunc[0] <= v <= trunc[1]:
            break
    return v


def rng_laplace(lambd=1, trunc=None):
    """
    Generate random numbers from a Laplace distribution

    Parameters
    ----------
    lambd: float
        The scale of the distribution
    trunc: None, tuple
        Specifies whether the distribution is truncated. If it's
        not None then it must be a 2-tuple indicating the range where
        the distribution is defined.

    Returns
    -------
    float
        A Laplace distributed number, optionally in a
        specific range if the distribution is truncated.

    """
    while True:
        while True:
            u = 2 * _rnd.random() - 1
            if u != 0:
                break
        s = -1 if u < 0 else 1
        v = -s * lambd * _math.log(s * u)
        if not trunc or trunc[0] <= v <= trunc[1]:
            break
    return v


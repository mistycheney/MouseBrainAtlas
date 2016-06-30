"""Functions related to Lie Group."""

import numpy as np

def matrix_exp(w):
    """Return the exponential map for rotation 3-vector w."""
    wx, wy, wz = w
    w_skew = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])

    theta = np.sqrt(np.sum(w**2))

    exp_w = np.eye(3) + np.sin(theta)/theta*w_skew + (1-np.cos(theta))/theta**2*np.dot(w_skew, w_skew)
    return exp_w

def matrix_exp_v(v):
    """Return the exponenial map for translation + rotation 6-vector v."""
    t = v[:3]
    w = v[3:]

    theta = np.sqrt(np.sum(w**2))

    wx, wy, wz = w
    w_skew = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])
    exp_w = np.eye(3) + np.sin(theta)/theta*w_skew + (1-np.cos(theta))/(theta**2)*np.dot(w_skew, w_skew)

    V = np.eye(3) + (1-np.cos(theta))/(theta**2)*w_skew + (theta-np.sin(theta))/(theta**3)*np.dot(w_skew, w_skew)

    return exp_w, np.dot(V, t)

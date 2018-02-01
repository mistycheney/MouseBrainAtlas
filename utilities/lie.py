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
    """
    Compute the exponenial map for v = [t w]^T where v is the 6-vector of coordinates in the Lie algebra se(3), 
    comprising of two separate 3-vectors: w, the vector that determine rotation, and t, which determines translation.
    
    Args:
        v ((6,)-ndarray): the vector composed of translation parameters and rotation parameters.
        
    Returns:
        (exp_w_skew, Vt). The manifold element is a 4-by-4 matrix (exp_w_skew & Vt \\ 0 1).
        
    """
    t = v[:3] # translation components
    w = v[3:] # rotation components

    # epsilon = 1e-8
    epsilon = 0
    theta = np.sqrt(np.sum(w**2)) + epsilon # Is this way to resolve divide-by-zero problem correct?
    
    wx, wy, wz = w
    w_skew = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])
    exp_w = np.eye(3) + np.sin(theta)/theta*w_skew + (1-np.cos(theta))/(theta**2)*np.dot(w_skew, w_skew)

    V = np.eye(3) + (1-np.cos(theta))/(theta**2)*w_skew + (theta-np.sin(theta))/(theta**3)*np.dot(w_skew, w_skew)

    return exp_w, np.dot(V, t)

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline


# Assume you have N 2D points (x_t, y_t), and t is time or frame index
def fit_spline(positions):
    t = np.arange(len(positions))  # could also use actual timestamps
    positions = np.array(positions)  # shape (N, 2)
    x = positions[:, 0]
    y = positions[:, 1]

    # Fit cubic splines to x(t) and y(t)
    # spline_x = CubicSpline(
    # t, x, bc_type='natural')  # or 'clamped' for custom derivative
    # spline_y = CubicSpline(t, y, bc_type='natural')

    spline_x = UnivariateSpline(t, x)  # s is a smoothing factor
    spline_y = UnivariateSpline(t, y)
    return spline_x, spline_y, t

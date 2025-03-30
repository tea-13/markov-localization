import numpy as np
from scipy.signal import convolve, convolve2d

from .type_alias import Color, Point

def curicular_conv(mat, kernel, is_2d=True):
    """function for curicular convolution.

    Args:
        mat: matrix for convolution.
        kernel: convolution kernel.
        is_2d: one-dimensional or two-dimensional convolution.

    Returns:
        matrix with applied convolution.

    """
    mat = np.pad(mat, pad_width=1, mode="wrap")

    if is_2d:
        mat = convolve2d(mat, kernel, mode="valid")
    else:
        mat = convolve(mat, kernel, mode="valid")

    mat /= np.sum(mat)
    return mat


def intepolate_color(t: float) -> Color:
    """function for color interpolation.

    Args:
        t: linear interpolation parameter.

    Returns:
        interpolated color.

    """
    red = np.array([255, 0, 0])
    green = np.array([0, 255, 0])
    yellow = np.array([255, 255, 0])

    # color = t*green + (1-t)*red  # linear interpolation
    if t < 0.5:
        c = t*yellow + (0.5-t)*red
    else:
        c = (t-0.5)*green + (1-t)*yellow

    return tuple(map(int, c))


def rotate_matrix(alpha: float, degree: bool = True) -> np.array:
    """function for color interpolation.

    Args:
        alpha: rotation angle.
        degree: in degrees or radians.

    Returns:
        rotate matrix.

    """
    
    if degree:
        alpha = np.deg2rad(alpha)

    mt = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)],
    ])

    return mt


def calc_dist_2d(p1: Point, p2: Point):
    """function to calculate the distance between two points.

    Args:
        p1: point 1.
        p2: point 2.

    Returns:
        distance between points.

    """

    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
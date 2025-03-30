import numpy as np

from enum import Enum

class RobotOrientation(Enum):
    """Enumeration class of possible robot orientations
    """
    Up = 0
    Down = 1
    Left = 2
    Right = 3


class Movement(Enum):
    """Class enumerating possible robot movements
    """
    FORWARD = 0
    BACKWARD = 1
    TURN_CW = 2
    TURN_CCW = 3


DIR1 = {
    Movement.FORWARD: np.array([1, 0]),
    Movement.BACKWARD: np.array([-1, 0]),
}

DIR2 = {
    Movement.FORWARD: np.array([0, -1]),
    Movement.BACKWARD: np.array([0, 1]),
}

DIR3 = {
    Movement.FORWARD: np.array([-1, 0]),
    Movement.BACKWARD: np.array([1, 0]),
}

DIR4 = {
    Movement.FORWARD: np.array([0, 1]),
    Movement.BACKWARD: np.array([0, -1]),
}

ROBOT_DIR_ORIENT = {
    RobotOrientation.Right: DIR1,
    RobotOrientation.Up: DIR2,
    RobotOrientation.Left: DIR3,
    RobotOrientation.Down: DIR4,
}

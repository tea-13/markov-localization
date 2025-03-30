import pygame as pg
import numpy as np

from typing import Dict

from .type_alias import Color

# ============== Settings ================
FPS: int = 30
DEBUG_MODE = False

# Display height and width
WINDOW_HEIGHT: int = 800
WINDOW_WIDTH: int = 1280

# Grid settings
GRID_SIZE: int = 600
CELL_SIZE: int = 40
CELL_NUM: int = int(GRID_SIZE // CELL_SIZE)

# Robot settings
ROBOT_SIZE: int = CELL_SIZE - 7

# Font settings
CELL_FONT = pg.font.SysFont('arial', 12)
TITLE_FONT1 = pg.font.SysFont('arial', 30)
TITLE_FONT2 = pg.font.SysFont('arial', 20)
FONT_COLOR: Color = (255, 255, 255)

# Probability settings
PROB_SENSOR_TRUE = 0.8  # Probability of correct measurement
PROB_SENSOR_FALSE = 0.2  # Probability of measurement error
PROB_TURN = 0.8

PROB_MOVE_X = np.array([
    [0.0, 0.0, 0.0],
    [0.1, 0.8, 0.1],
    [0.0, 0.0, 0.0],
])

PROB_MOVE_Y = np.array([
    [0.0, 0.1, 0.0],
    [0.0, 0.8, 0.0],
    [0.0, 0.1, 0.0],
])

PROB_TURN_CONV = np.array([0.1, 0.8, 0.1])

# Color settings
BACKGROUND_COLOR: Color = (164, 164, 164)
ROBOT_COLOR: Color = (255, 0, 255)
BORDER_COLOR: Color = (100, 100, 100)
TITLE_COLOR: Color = (90, 90, 255)

COLOR_MAP: Dict[int, Color] = {
    0: (50, 50, 50),
    1: (20, 20, 20),
    2: (106, 255, 0),
    3: (0, 255, 200),
    4: (0, 10, 255),
    5: (255, 150, 0),
}
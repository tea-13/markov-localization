import pygame as pg
import numpy as np
import cv2

from abc import ABC, abstractmethod

from .config import *
from .utils import intepolate_color, calc_dist_2d
from .type_alias import Color, Point, Offset


class GridCell:
    """Class that stores information and determines how a cell is displayed on the screen.
    
    Attributes:
        rect (pygame.Rect): Concrete cell.
        color (Tuple[int, int, int]): Cell color.
        prob (float): Cell probability.

    """

    def __init__(self):
        self.rect = None
        self.color: Color = None
        self.border_color: Color = None
        self.color_num: int = None
        self.prob: float = 0.0

    def draw_prob_cell(self, screen):
        """Method for drawing a cell in a probability grid

        Args:
            screen: pygame display.

        """
        pg.draw.rect(screen, self.color, self.rect)
        CELL_FONT.set_bold(True)
        text_surface = CELL_FONT.render(f"{self.prob: .3f}", True, FONT_COLOR)
        screen.blit(text_surface, (self.rect.x, self.rect.centery))

    def draw_sim_cell(self, screen):
        """Method for drawing a cell in a simulation grid

        Args:
            screen: pygame display.

        """
        if self.color is not None:
            pg.draw.rect(screen, self.color, self.rect)
        pg.draw.rect(screen, self.border_color, self.rect, 1)


class AbstractMap(ABC):
    """Abstract class for map
    
    """
    @abstractmethod
    def __init__(self, offset: Offset):
        pass
    
    @abstractmethod
    def create_grid(self):
        pass
    
    @abstractmethod
    def render_grid(self, screen):
        pass


class ProbabilityMap(AbstractMap):
    """Class for interacting with a probability map.
    
    Attributes:
        offset (Offset): Grid offset.
        probability_map (np.array): Numpy array stores data about a grid.
        map (List): The list stores the grid cells..

    """
    def __init__(self, offset: Offset = (0, 0)):
        self.offset = offset
        self.probability_map = np.full((CELL_NUM, CELL_NUM), 
                1/(CELL_NUM*CELL_NUM), dtype=np.float32)
        self.map = [[None for _ in range(CELL_NUM)] for _ in range(CELL_NUM)]

        self.create_grid()

    def create_grid(self):
        """Method to fill an array containing grid cells

        """
        x_pad, y_pad = self.offset

        for y in range(CELL_NUM):
            for x in range(CELL_NUM):
                cell = GridCell()
                cell.rect = pg.Rect(x*(CELL_SIZE + 1) + x_pad, y*(CELL_SIZE + 1) + y_pad, CELL_SIZE, CELL_SIZE)
                cell.prob = self.probability_map[y, x]
                cell.color = intepolate_color(cell.prob)

                self.map[y][x] = cell

    def render_grid(self, screen):
        """Method for displaying a grid on the screen

        Args:
            screen: pygame display.

        """
        for y in range(CELL_NUM):
            for x in range(CELL_NUM):
                self.map[y][x].prob = self.probability_map[y, x]
                
                if (np.max(self.probability_map) - self.probability_map[y, x]) < 1e-3 and DEBUG_MODE:
                    self.map[y][x].color = intepolate_color(1)
                else:
                    self.map[y][x].color = intepolate_color(self.map[y][x].prob)

                self.map[y][x].draw_prob_cell(screen)
    
    def sensor_update(self, mat: np.array):
        """Method for update probability matrix

        Args:
            mat: new matrix.

        """
        if mat is not None:
            self.probability_map = mat
    

class SimulationMap(AbstractMap):
    """Class for interacting with a probability map.
    
    Attributes:
        offset (Offset): Grid offset.
        sim_map (np.array): Numpy array stores data about a grid.
        map (List): The list stores the grid cells.

    """

    def __init__(self, offset: Offset = (0, 0), path="map.png"):
        self.offset = offset
        # self.sim_map = np.full((CELL_NUM, CELL_NUM), 0, dtype=np.int32)
        self.sim_map = SimulationMap.read_map(path)
        self.map = [[None for _ in range(CELL_NUM)] for _ in range(CELL_NUM)]
        self.background_rect = pg.Rect(offset[0], offset[1], GRID_SIZE, GRID_SIZE)

        self.create_grid()

    @staticmethod
    def read_map(path):
        """Method for reading map file

        Args:
            path: path to file.

        """
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def create_grid(self):
        """Method to fill an array containing grid cells
        
        """
        x_pad, y_pad = self.offset

        for y in range(CELL_NUM):
            for x in range(CELL_NUM):
                cell = GridCell()
                cell.rect = pg.Rect(x*CELL_SIZE + x_pad, y*CELL_SIZE + y_pad, CELL_SIZE, CELL_SIZE)
                cell.border_color = COLOR_MAP[0]
                cell.color_num = self.sim_map[y, x]
                
                if self.sim_map[y, x] != 0:
                    cell.color = COLOR_MAP[self.sim_map[y, x]]

                self.map[y][x] = cell

    def render_grid(self, screen):
        """Method for displaying a grid on the screen

        Args:
            screen: pygame display.

        """
        pg.draw.rect(screen, BORDER_COLOR, self.background_rect)
        for y in range(CELL_NUM):
            for x in range(CELL_NUM):
                self.map[y][x].draw_sim_cell(screen)

    def create_sensor_mt(self, color_num: int, p_hit: float, p_miss: float) -> np.array:
        """Method of creating a sensor matrix

        Args:
            color_num (int): color number.
            p_hit (float): probability of correct sensor operation
            p_miss (float): probability of sensor malfunction
        
        Return:
            probability sensor matrix

        """

        if color_num == 0:
            mt = np.full(self.sim_map.shape, 1.0, dtype=np.float32)
            return mt
        
        mt = np.full(self.sim_map.shape, p_miss, dtype=np.float32)

        for y in range(CELL_NUM):
            for x in range(CELL_NUM):
                if self.sim_map[y][x] == color_num:
                    mt[y][x] = p_hit

        return mt

    def get_center_rect(self, x: int, y: int) -> Point:
        """Method for obtaining the center of the cell

        Args:
            x (int): row.
            y (int): column.
        
        Return:
            cell center

        """
        origin_rect = self.map[y][x]
        return origin_rect.rect.center

    def get_color_num(self, x: int, y: int) -> int:
        """Method for obtaining the color number

        Args:
            x (int): row.
            y (int): column.
        
        Return:
            color number

        """
        return self.map[y][x].color_num
    
    def get_nearest_cell(self, x: int, y: int) -> Point:
        """Method for obtaining the nearest cell

        Args:
            x (int): abscissa.
            y (int): ordinate.
        
        Return:
            nearest cell

        """
        min_dsit = GRID_SIZE
        nearest_cell_xy = None
        p2 = [x, y]

        for yy in range(CELL_NUM):
            for xx in range(CELL_NUM):
                p1 = self.get_center_rect(xx, yy)
                dist = calc_dist_2d(p1, p2)
                
                if dist < min_dsit:
                    min_dsit = dist
                    nearest_cell_xy = [xx, yy]

        return nearest_cell_xy

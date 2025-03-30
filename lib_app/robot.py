import pygame as pg
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from .utils import curicular_conv, rotate_matrix
from .movment_orient import RobotOrientation, Movement
from .config import *


@dataclass
class Sensor:
    """Dataclass for snesor.
    
    Attributes:
        p_hit (float): probability of correct sensor operation
        p_miss (float): probability of sensor malfunction
        

    """
    p_hit: float
    p_miss: float


class AbstractRobot(ABC):
    """Abstract class for robot
    """
    @abstractmethod
    def move(self, direction_mag, sim_map, prob_map):
        pass
    
    @abstractmethod
    def rotate(self, orient, sim_map, prob_map):
        pass
    
    @abstractmethod
    def render(self, screen):
        pass


class Robot(AbstractRobot):
    """Сlass that defines the behavior of a robot.
    
    Attributes:
        xy (np.array): robot position.
        start_pos (np.array): robot start position.
        step_between_color (int): step between two color.
        orientation (int): orientation in degrees.
        start_orient_color_num (int): start orient color number.
        end_orient_color_num (int): end orient color number.
        action_list (List[Movement]): list all movment between two color.
        start_orientation (bool): start orientation algorithm.
        localization_loop (bool): start localization loop.
        orientation_pose (np.array): current robot pose.
        p_rotate (np.array): vector of robot orientation probabilities.
        sensor (Sensor): sensor.
        rect (pg.Rect): orientation rect.

    """

    def __init__(self, x, y, orient, pad, rect_size):
        self.xy = np.array([x, y])
        self.start_pos = self.xy
        
        self.step_between_color: int = 0
        self.orientation: int = orient
        self.start_orient_color_num: int = 0
        self.end_orient_color_num: int = 0

        self.action_list: List[Movement] = []

        self.start_orientation: bool = False
        self.localization_loop: bool = False
        
        self.orientation_pose = [np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]
        self.p_rotate = np.array([0.25, 0.25, 0.25, 0.25])

        self.sensor = Sensor(PROB_SENSOR_TRUE, PROB_SENSOR_FALSE)
        x_pad, y_pad = pad
        self.rect = pg.Rect(x_pad, y_pad, rect_size, rect_size)

        self.calculate_orientation()

    def move(self, direction_mag: int, sim_map, prob_map):
        """Method for move robot

        Args:
            direction_mag (int): direction and magnitude of movement
            sim_map (SimulationMap): simulation map.
            prob_map (ProbabilityMap): probability map

        """
        dir_vec = direction_mag*np.array([1, 0])
        move_vec = np.int32(np.dot(rotate_matrix(self.orientation), dir_vec))
        # print(move_vec)
        
        temp_pos = self.xy + move_vec

        dx = temp_pos[0] - self.start_pos[0]
        dy = temp_pos[1] - self.start_pos[1]

        if abs(dx) >= GRID_SIZE and dx > 0:
            temp_pos = self.xy - np.array([GRID_SIZE-CELL_SIZE, 0])
        elif abs(dx) <= CELL_SIZE and dx < 0:
            temp_pos = self.xy + np.array([GRID_SIZE-CELL_SIZE, 0])
        elif abs(dy) >= GRID_SIZE and dy > 0:
            temp_pos = self.xy - np.array([0, GRID_SIZE-CELL_SIZE])
        elif abs(dy) <= CELL_SIZE and dy < 0:
            temp_pos = self.xy + np.array([0, GRID_SIZE-CELL_SIZE])

        self.xy = temp_pos

        self.calculate_orientation()
        
        action = Movement.FORWARD if direction_mag > 0 else Movement.BACKWARD

        if self.start_orientation:
            self.step_between_color += 1
            self.action_list.append(action)
        
        return self.localization2(sim_map, prob_map, action)

    def rotate(self, orient: int, sim_map, prob_map):
        """Method for rotate robot

        Args:
            orient (int): the amount of rotation of the robot
            sim_map (SimulationMap): simulation map.
            prob_map (ProbabilityMap): probability map

        """
        self.orientation += orient

        if abs(self.orientation) % 360 == 0:
            self.orientation = 0

        self.calculate_orientation()
        action = Movement.TURN_CW if orient > 0 else Movement.TURN_CCW

        if self.start_orientation:
            self.step_between_color += 1
            self.action_list.append(action)

            if action == Movement.TURN_CW:
                step = -1
            elif action == Movement.TURN_CCW:
                step = 1
    
            self.p_rotate = np.roll(self.p_rotate, step)
            self.p_rotate = curicular_conv(self.p_rotate, PROB_TURN_CONV, is_2d=False)

        # self.localization()

    def calculate_orientation(self):
        """Method for calculating orientation for robot drawing

        Args:
            sim_map (SimulationMap): simulation map.

        """
        front_point = np.array([ROBOT_SIZE//2, 0])
        side_point1 = np.array([-ROBOT_SIZE//2, ROBOT_SIZE//2])
        side_point2 = np.array([-ROBOT_SIZE//2, -ROBOT_SIZE//2])

        self.orientation_pose[0] = np.dot(rotate_matrix(self.orientation), front_point)
        self.orientation_pose[1] = np.dot(rotate_matrix(self.orientation), side_point1)
        self.orientation_pose[2] = np.dot(rotate_matrix(self.orientation), side_point2)

        for i in range(3):
            self.orientation_pose[i] += self.xy

    def calc_orient_with_color(self, sim_map):
        """Method for calculating all possible orientations between two colors

        Args:
            sim_map (SimulationMap): simulation map.

        """

        global ROBOT_DIR_ORIENT
        print(f"start: {self.start_orient_color_num}, end: {self.end_orient_color_num}, step: {self.step_between_color}")
        print(f"action: {self.action_list}")

        grid = sim_map.sim_map.copy()
        path = self.action_list
        
        height, width = len(grid), len(grid[0])  # Размер карты
        
        # 1. Найти все цветные клетки на карте
        color_start_positions = []
        color_end_positions = []
        for x in range(height):
            for y in range(width):
                if grid[x][y] == self.start_orient_color_num:
                    color_start_positions.append((x, y, grid[x][y]))  # (координаты, цвет)
                if grid[x][y] == self.end_orient_color_num:
                    color_end_positions.append((x, y, grid[x][y]))  # (координаты, цвет)
        
        # 2. Перебрать все возможные пары цветных клеток
        directions = [RobotOrientation.Down, RobotOrientation.Left, RobotOrientation.Up, RobotOrientation.Right]
        num_end_dir = {
            RobotOrientation.Left: 0,
            RobotOrientation.Down: 0, 
            RobotOrientation.Right: 0, 
            RobotOrientation.Up: 0,
        }
        all_num_dir = 0

        for x1, y1, color1 in color_start_positions:
            for x2, y2, color2 in color_end_positions:
                
                # 3. Проверяем, можно ли пройти путь от (x1, y1) к (x2, y2) с учетом цикличности и направлений
                for start_direction in range(4):  # Пробуем все начальные направления (0, 1, 2, 3)
                    cur_x, cur_y = x1, y1
                    cur_dir = start_direction  # Начальное направление
                    
                    for move in path:
                        if move == Movement.TURN_CCW:  # Поворот влево
                            cur_dir = (cur_dir + 1) % 4
                            continue

                        elif move == Movement.TURN_CW:  # Поворот вправо
                            cur_dir = (cur_dir - 1) % 4
                            continue
                        
                        direct = directions[cur_dir]
                        dx, dy = ROBOT_DIR_ORIENT[direct][move]
                        cur_x = (cur_x + dx) % height
                        cur_y = (cur_y + dy) % width
                    
                    # 4. Если финальная точка совпадает с (x2, y2), значит, (x1, y1) — стартовая
                    if cur_x == x2 and cur_y == y2:
                        d = directions[cur_dir]
                        num_end_dir[d] += 1
                        all_num_dir += 1

        num_end_dir[RobotOrientation.Left] /= all_num_dir
        num_end_dir[RobotOrientation.Up] /= all_num_dir
        num_end_dir[RobotOrientation.Down] /= all_num_dir
        num_end_dir[RobotOrientation.Right] /= all_num_dir

        num_end_dir[RobotOrientation.Left] *= PROB_TURN
        num_end_dir[RobotOrientation.Up] *= PROB_TURN
        num_end_dir[RobotOrientation.Down] *= PROB_TURN
        num_end_dir[RobotOrientation.Right] *= PROB_TURN

        return num_end_dir

    def start_localization(self, flag: bool):
        """Method for start localization

        Args:
            flag (bool): start localization?

        """

        self.localization_loop = flag

    def localization2(self, sim_map, prob_map, action: Movement):
        """Method for localization with probabilistic orientation

        Args:
            sim_map (SimulationMap): simulation map.
            prob_map (ProbabilityMap): probability map
            actiom (Movement): robot action list

        """
        
        if not self.localization_loop:
            return None

        step, axis = 0, 0
        conv = np.zeros((3, 3))
        
        p_map1 = prob_map.probability_map.copy()
        p_map2 = prob_map.probability_map.copy()
        p_map3 = prob_map.probability_map.copy()
        p_map4 = prob_map.probability_map.copy()

        axis = 1
        conv = PROB_MOVE_X
            
        if action == Movement.FORWARD:
            step = 1
        elif action == Movement.BACKWARD:
            step = -1
        
        p_map1 = np.roll(p_map1, step, axis=axis)
        new_p_map1 = curicular_conv(p_map1, conv)

        if action == Movement.FORWARD:
            step = -1
        elif action == Movement.BACKWARD:
            step = 1
        
        p_map2 = np.roll(p_map2, step, axis=axis)
        new_p_map2 = curicular_conv(p_map2, conv)
            
        axis = 0
        conv = PROB_MOVE_Y

        if action == Movement.FORWARD:
            step = -1
        elif action == Movement.BACKWARD:
            step = 1

        p_map3 = np.roll(p_map3, step, axis=axis)
        new_p_map3 = curicular_conv(p_map3, conv)

        if action == Movement.FORWARD:
            step = 1
        elif action == Movement.BACKWARD:
            step = -1

        p_map4 = np.roll(p_map4, step, axis=axis)
        new_p_map4 = curicular_conv(p_map4, conv)


        nearest_cell_xy = sim_map.get_nearest_cell(*self.xy)
        color_num = sim_map.get_color_num(*nearest_cell_xy)

        if color_num != 0:
            if self.start_orientation:
                self.end_orient_color_num = color_num
                direct = self.calc_orient_with_color(sim_map)
                p = [
                    direct[RobotOrientation.Down],
                    direct[RobotOrientation.Left],
                    direct[RobotOrientation.Up],
                    direct[RobotOrientation.Right],
                ]
                print(f"{direct}")

                if np.max(self.p_rotate) < np.max(p):
                    self.p_rotate[0] = p[0]
                    self.p_rotate[1] = p[1]
                    self.p_rotate[2] = p[2]
                    self.p_rotate[3] = p[3]
                    self.p_rotate /= np.sum(self.p_rotate)

                self.start_orient_color_num = color_num
                self.step_between_color = 0
                self.end_orient_color_num = 0
                self.action_list = []
            else:
                self.start_orientation = True
                self.start_orient_color_num = color_num
                self.step_between_color = 0
                self.action_list = []
    

        mat = sim_map.create_sensor_mt(color_num, 
                self.sensor.p_hit, self.sensor.p_miss)
        # self.prob_map.sensor_update(mt)
                    
        # print(f"{nearest_cell_xy= }; {color_num= }")

        mat *= self.p_rotate[0]*new_p_map1 + self.p_rotate[2]*new_p_map2 + self.p_rotate[1]*new_p_map3 + self.p_rotate[3]*new_p_map4
        mat /= np.sum(mat)

        return mat
    
    def localization(self, sim_map, prob_map, action: Movement):
        """Method for localization with clear orientation

        Args:
            sim_map (SimulationMap): simulation map.
            prob_map (ProbabilityMap): probability map
            actiom (Movement): robot action list

        """
        if not self.localization_loop:
            return None

        p_map = prob_map.probability_map.copy()
        step, axis = 0, 0
        conv = np.zeros((3, 3))

        if self.orientation == 0:
            axis = 1
            conv = PROB_MOVE_X
            
            if action == Movement.FORWARD:
                step = 1
            elif action == Movement.BACKWARD:
                step = -1

        elif np.abs(self.orientation) == 180:
            axis = 1
            conv = PROB_MOVE_X
            
            if action == Movement.FORWARD:
                step = -1
            elif action == Movement.BACKWARD:
                step = 1
            
        elif self.orientation == -90 or self.orientation == 270:
            axis = 0
            conv = PROB_MOVE_Y

            if action == Movement.FORWARD:
                step = -1
            elif action == Movement.BACKWARD:
                step = 1

        elif self.orientation == 90 or self.orientation == -270:
            axis = 0
            conv = PROB_MOVE_Y 

            if action == Movement.FORWARD:
                step = 1
            elif action == Movement.BACKWARD:
                step = -1

        
        p_map = np.roll(p_map, step, axis=axis)
        new_p_map = curicular_conv(p_map, conv)

        nearest_cell_xy = sim_map.get_nearest_cell(*self.xy)
        color_num = sim_map.get_color_num(*nearest_cell_xy)

        mat = sim_map.create_sensor_mt(color_num, 
                self.sensor.p_hit, self.sensor.p_miss)
        # self.prob_map.sensor_update(mt)
                    
        print(f"{nearest_cell_xy= }; {color_num= }")

        mat *= new_p_map
        mat /= np.sum(mat)

        return mat

    def update_sensor_data(self, sim_map):
        """Method for create sensor probability map

        Args:
            sim_map (SimulationMap): simulation map.

        """
        
        nearest_cell_xy = sim_map.get_nearest_cell(*self.xy)
        color_num = sim_map.get_color_num(*nearest_cell_xy)
        mt = sim_map.create_probability_mt(color_num, 
                self.sensor.p_hit, self.sensor.p_miss)

        print(f"{nearest_cell_xy= }; {color_num= }")
        
        return mt

    def render(self, screen):
        """Method for displaying robot on the screen

        Args:
            screen: pygame display.

        """

        pg.draw.rect(screen, BORDER_COLOR, self.rect, 1)
        
        CELL_FONT.set_bold(True)
        text_prob1 = CELL_FONT.render(f"{self.p_rotate[0]: .2f}", True, FONT_COLOR)
        text_prob2 = CELL_FONT.render(f"{self.p_rotate[1]: .2f}", True, FONT_COLOR)
        text_prob3 = CELL_FONT.render(f"{self.p_rotate[2]: .2f}", True, FONT_COLOR)
        text_prob4 = CELL_FONT.render(f"{self.p_rotate[3]: .2f}", True, FONT_COLOR)

        screen.blit(text_prob1, (self.rect.right - text_prob1.get_width(), self.rect.centery - text_prob1.get_height()//2))
        screen.blit(text_prob2, (self.rect.centerx - text_prob1.get_width()//2, self.rect.y))
        screen.blit(text_prob3, (self.rect.x, self.rect.centery - text_prob1.get_height()//2))
        screen.blit(text_prob4, (self.rect.centerx - text_prob1.get_width()//2, self.rect.bottom - text_prob1.get_height()))

        pg.draw.polygon(screen, ROBOT_COLOR, self.orientation_pose)

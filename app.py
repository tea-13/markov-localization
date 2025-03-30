import pygame as pg
import numpy as np

# ============== PyGame =================
pg.init()
pg.font.init()

from lib_app import AppTitle
from lib_app import ProbabilityMap, SimulationMap
from lib_app import Robot
from lib_app import Singleton
from lib_app import (
    WINDOW_WIDTH, 
    WINDOW_HEIGHT, 
    GRID_SIZE, 
    BACKGROUND_COLOR, 
    CELL_SIZE,
    FPS,
)


class App(Singleton):
    """Singleton class responsible for the functionality of the application.
    
    Attributes:
        run_app (bool): Application running flag.
        clock (pygame.time.Clock): pygame clock.
        screen (pygame.display): pygame display.
        prob_map (ProbabilityMap): Probability map.
        sim_map (SimulationMap): Simulation map.

    """
    def __init__(self):
        self.run_app = True
        self.auto_sensor = False
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT), pg.NOFRAME)

        pad = (2*int((WINDOW_WIDTH - 2*GRID_SIZE) / 3)+GRID_SIZE, int((WINDOW_HEIGHT - GRID_SIZE) / 2))
        self.prob_map = ProbabilityMap(offset=pad)
        
        pad = (int((WINDOW_WIDTH - 2*GRID_SIZE) / 3), int((WINDOW_HEIGHT - GRID_SIZE) / 2))
        self.sim_map = SimulationMap(offset=pad)

        pad = (50, int((WINDOW_HEIGHT - GRID_SIZE)) / 8)
        self.title = AppTitle(offset=pad)

        self.robot_pos = [0, 0]
        c_x, x_y = self.sim_map.get_center_rect(*self.robot_pos)
        self.robot = Robot(c_x, x_y, 0, pad=(1110, 10), rect_size=80)

    def __del__(self):
        """Class destructor
        """
        pg.quit()

    def run(self):
        """Application launch method
        
        """
        while self.run_app:
            # Events processing
            for event in pg.event.get():
                self.evants_handler(event)

            # Fill the background with gray
            self.render_all()
            
            # FPS
            self.clock.tick(FPS)

            # Flip the display
            pg.display.flip()

    def render_all(self):
        """Method of displaying graphical applications
        
        """
        self.screen.fill(BACKGROUND_COLOR)
        self.title.rendre_text(self.screen)
        self.prob_map.render_grid(self.screen)
        self.sim_map.render_grid(self.screen)
        self.robot.render(self.screen)
    
    def evants_handler(self, event):
        """Event handling method
        
        Args:
            event: pygame event.

        """
        if event.type == pg.QUIT:
            self.run_app = False

        self.keyboards_handler(event)
        self.mouse_handler(event)

    def keyboards_handler(self, event):
        """keyboards handling method
        
        Args:
            event: pygame event.

        """
        global DEBUG_MODE

        if event.type == pg.KEYDOWN:
            input_key = event.key

            match input_key:
                case pg.K_ESCAPE:
                    self.run_app = False
                case pg.K_a:
                    self.robot.rotate(-90, self.sim_map, self.prob_map)
                case pg.K_d:
                    self.robot.rotate(90, self.sim_map, self.prob_map)
                case pg.K_w:
                    mt = self.robot.move(CELL_SIZE, self.sim_map, self.prob_map)
                    self.prob_map.sensor_update(mt)
                case pg.K_s:
                    mt = self.robot.move(-CELL_SIZE, self.sim_map, self.prob_map)
                    self.prob_map.sensor_update(mt)
                case pg.K_c:
                    self.auto_sensor = not self.auto_sensor
                    self.robot.start_localization(self.auto_sensor)
                case pg.K_z:
                    nearest_cell_xy = self.sim_map.get_nearest_cell(*self.robot.xy)
                    color_num = self.sim_map.get_color_num(*nearest_cell_xy)
                    mat = self.sim_map.create_sensor_mt(color_num, 
                            self.robot.sensor.p_hit, self.robot.sensor.p_miss)
                    mat *= self.prob_map.probability_map
                    mat /= np.sum(mat)
                    self.prob_map.sensor_update(mat)
                case pg.K_x:
                    DEBUG_MODE = not DEBUG_MODE
                    
            print(f"Нажата клавиша: {pg.key.name(input_key)}, {input_key}")
            print(f"Модификатор: {event.mod}")
            print(f"robot orient: {self.robot.orientation}")

    def mouse_handler(self, event):
        """Mouse handling method
        
        Args:
            event: pygame event.

        """
        if event.type == pg.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            mouse_button = event.button
            print(f"Позиция мыши: {mouse_pos}")
            print(f"Идентификатор кнопки мыши: {mouse_button}")


if __name__ == "__main__":
    app = App()
    app.run()
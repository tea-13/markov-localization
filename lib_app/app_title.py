from .config import *
from .type_alias import Offset

class AppTitle:
    """Class for displaying title.
    
    Attributes:
        offset (Offset): indent for drawing.

    """
    
    def __init__(self, offset: Offset):
        self.offset = offset
        self.main_title = TITLE_FONT1.render("Markov localization", True, TITLE_COLOR)
        self.key_bind = TITLE_FONT2.render("W/S - forward/backward; A/D - turn left/right; C - start simulation; Z - sencor update; X - change view mode; ESC - exit", True, TITLE_COLOR)

    def rendre_text(self, screen):
        """Method for displaying a text on the screen

        Args:
            screen: pygame display.

        """
        screen.blit(self.main_title, self.offset)
        screen.blit(self.key_bind, (self.offset[0], self.offset[1]+self.main_title.get_height()))
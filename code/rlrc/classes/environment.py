import pygame

from ..constants.colors import BLACK


class Environment:
    def __init__(self, walls):
        # Walls defined as line segments: (x1,y1)-(x2,y2)
        self.walls = walls

    def draw(self, surface):
        for wall in self.walls:
            pygame.draw.line(
                surface, 
                BLACK,                      # color
                (wall[0], wall[1]),         # starting point
                (wall[2], wall[3]),         # ending point
                2                           # thickness
            )

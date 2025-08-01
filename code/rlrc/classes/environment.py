import pygame

from ..constants.colors import BLACK


class Environment:
    def __init__(self):
        # Walls defined as line segments: (x1,y1)-(x2,y2)
        # I quattro muri perimetrali, definiti come segmenti (x1,y1,x2,y2):
        from ..constants.robot import MAP_GRID_SIZE
        self.x_start = MAP_GRID_SIZE // 2 - 300
        self.x_end = MAP_GRID_SIZE // 2 + 300
        self.y_start = MAP_GRID_SIZE // 2 - 200
        self.y_end = MAP_GRID_SIZE // 2 + 200
        self.walls = [
            (self.x_start, self.y_start, self.x_end, self.y_start),               # muro superiore
            (self.x_start, self.y_start, self.x_start, self.y_end),      # muro a destra
            (self.x_start, self.y_end, self.x_end, self.y_end),     # muro inferiore
            (self.x_end, self.y_start, self.x_end, self.y_end)               # muro a sinistra
        ]

    def draw(self, surface):
        for wall in self.walls:
            pygame.draw.line(
                surface, 
                BLACK,                      # color
                (wall[0], wall[1]),         # starting point
                (wall[2], wall[3]),         # ending point
                2                           # thickness
            )


# class Environment:
#     def __init__(self, width, height):
#         self.width = width
#         self.height = height
#         # Walls defined as line segments: (x1,y1)-(x2,y2)
#         # I quattro muri perimetrali, definiti come segmenti (x1,y1,x2,y2):
#         offset = 100
#         self.walls = [
#             (0, 0, width, 0),               # muro superiore
#             (width, 0, width, height),      # muro a destra
#             (width, height, 0, height),     # muro inferiore
#             (0, height, 0, 0)               # muro a sinistra
#         ]

#     def draw(self, surface):
#         for wall in self.walls:
#             pygame.draw.line(
#                 surface, 
#                 BLACK,                      # color
#                 (wall[0], wall[1]),         # starting point
#                 (wall[2], wall[3]),         # ending point
#                 2                           # thickness
#             )
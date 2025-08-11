import pygame
import math
import numpy as np

from ..constants.configuration import LABELS_INT_TO_STR, LABELS_STR_TO_INT
from ..constants.colors import BLACK, BLUE_SEMITRASPARENT, BLUE_SEMITRASPARENT_DARK, GRAY, WHITE, BLUE, GREEN, RED, TMP_BACKGROUND


class Graphics:
            
    def __init__(self, envirnonment, robot):
        self.environment = envirnonment
        self.robot = robot
        pygame.init()
        pygame.display.set_caption("Robot Vacuum Prototype")
        self.font = pygame.font.Font(None, 20)
        self.screen = pygame.display.set_mode((envirnonment.w*2 + 150, envirnonment.h + 100))
        self.clock = pygame.time.Clock()
                
    def update(self, environment, robot, rays, status, score):
        grid_status, battery = status
        self.screen.fill(TMP_BACKGROUND)
        # Left: real environment
        real_world_surface = self.screen.subsurface((50, 50, environment.w, environment.h))
        real_world_surface.fill(WHITE)
        self._draw_walls(real_world_surface)
        self._draw_robot(real_world_surface)
        self._draw_lidar(real_world_surface, rays)
        # Right: internal map
        internal_map_surface = self.screen.subsurface((environment.w + 100, 50, environment.w, environment.h))
        internal_map_surface.fill(WHITE)
        self._draw_map(internal_map_surface, robot)
        self._draw_robot(internal_map_surface)
        self._draw_vision(internal_map_surface, environment, robot)
        

        # grid status
        for i, (val, count) in enumerate(sorted(grid_status.items())):
            txt = self.font.render(f'{LABELS_INT_TO_STR[int(val)]}: {count}', True, BLACK)
            self.screen.blit(txt, (environment.w + 150, 35 + 20*i))

        # battery
        txt = self.font.render(f'battery: {round(battery*100, 2)}%', True, BLACK)
        self.screen.blit(txt, (environment.w + 150, 35 + 20*len(grid_status)))

        # total reward
        txt = self.font.render(f'return (total reward): {score}', True, BLACK)
        self.screen.blit(txt, (environment.w + 150, 35 + 20*len(grid_status) + 20))

        # % clean over free
        clean_key = np.int16(LABELS_STR_TO_INT['clean'])
        if clean_key in grid_status:
            clean = grid_status[clean_key]
        else:
            clean = 0
        free = grid_status[np.int16(LABELS_STR_TO_INT['free'])]
        clean_over_free = round(clean / (clean + free) * 100, 2)
        txt = self.font.render(f'clean / (clean + free): {clean_over_free} %', True, BLACK)
        self.screen.blit(txt, (environment.w + 150, 35 + 20*len(grid_status) + 40))

        pygame.display.flip()       # Update all the screen
        self.clock.tick(60)              # ~60 FPS

    def _draw_walls(self, surface):
        for wall in self.environment.walls:
            pygame.draw.line(
                surface, 
                BLACK,                      # color
                (wall[0], wall[1]),         # starting point
                (wall[2], wall[3]),         # ending point
                2                           # thickness
            )
    
    def _draw_robot(self, surface):
        # Robot body
        pygame.draw.circle(surface, BLUE, (int(self.robot.x), int(self.robot.y)), self.robot.radius, 2)
        # Robot orientation
        orientation_x = int(self.robot.x + math.cos(self.robot.angle) * self.robot.radius * 1.5)
        orientation_y = int(self.robot.y + math.sin(self.robot.angle) * self.robot.radius * 1.5)
        pygame.draw.line(surface, BLUE, (int(self.robot.x), int(self.robot.y)), (orientation_x, orientation_y), 5)

    def _draw_lidar(self, surface, rays):
        for hit_x, hit_y in rays:
            if hit_x is not None and hit_y is not None:
                pygame.draw.line(surface, BLUE, (int(self.robot.x), int(self.robot.y)), (int(hit_x), int(hit_y)), 1)

    def _draw_vision(self, surface, environment, robot):
        # 1) Calcolo dimensioni e posizioni
        offset = 15.5 * robot.cell_side
        w = h = int(offset * 2)
        x_start = self.robot.x - offset
        y_start = self.robot.y - offset

        # Creo una surface temporanea che supporta il per-pixel alpha
        rect_surf = pygame.Surface((w, h), pygame.SRCALPHA)

        # Disegno il rettangolo pieno su rect_surf
        pygame.draw.rect(rect_surf, BLUE_SEMITRASPARENT_DARK, rect_surf.get_rect()) 

        # Blitto rect_surf sul surface principale, rispettando l’alpha
        surface.blit(rect_surf, (x_start, y_start))

        # left
        x = 0
        y = self.robot.y - robot.radius 
        w = self.robot.x - robot.radius
        h = robot.radius * 2
        rect_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(rect_surf, BLUE_SEMITRASPARENT, rect_surf.get_rect()) 
        surface.blit(rect_surf, (x, y))

        # right
        x = self.robot.x + robot.radius
        y = self.robot.y - robot.radius 
        w = environment.w
        h = robot.radius * 2
        rect_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(rect_surf, BLUE_SEMITRASPARENT, rect_surf.get_rect()) 
        surface.blit(rect_surf, (x, y))

        # up
        x = self.robot.x - robot.radius
        y = 0
        w = robot.radius * 2
        h = self.robot.y - robot.radius
        rect_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(rect_surf, BLUE_SEMITRASPARENT, rect_surf.get_rect()) 
        surface.blit(rect_surf, (x, y))

        # down
        x = self.robot.x - robot.radius
        y = self.robot.y + robot.radius
        w = robot.radius * 2
        h = environment.h
        rect_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(rect_surf, BLUE_SEMITRASPARENT, rect_surf.get_rect()) 
        surface.blit(rect_surf, (x, y))

    


    def _draw_map(self, surface, robot):
        # draw occupancy grid to a given surface of size ENVIRONMENT_WIDTH x ENVIRONMENT_HEIGHT
        for y in range(robot.h):
            for x in range(robot.w):
                val = self.robot.grid[y, x]
                if val == LABELS_STR_TO_INT['static obstacle']:
                    color = BLACK
                elif val == LABELS_STR_TO_INT['unknown']:
                    color = GRAY
                elif val == LABELS_STR_TO_INT['free']:
                    color = WHITE
                elif val == LABELS_STR_TO_INT['clean']:
                    color = GREEN
                elif val == LABELS_STR_TO_INT['base']:
                    color = BLUE
                else:
                    color = RED          # re-cleaned

                rect = pygame.Rect(x*robot.cell_side, y*robot.cell_side, robot.cell_side, robot.cell_side)
                pygame.draw.rect(surface, color, rect)

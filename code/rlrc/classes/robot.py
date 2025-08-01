import numpy as np
import math
import random
import pygame

from ..constants.robot import MAP_GRID_SIZE, CELL_SIDE
from ..constants.colors import YELLOW, BLACK, GRAY, WHITE, BLUE, SKY_BLUE, GREEN, RED


def ray_segment_intersection(px, py, dx, dy, x1, y1, x2, y2):
    """
    Calcola l'intersezione tra il raggio p + t*d e il segmento (x1,y1)-(x2,y2).
    Restituisce t (distanza lungo il raggio) e u (parametro sul segmento), oppure (None, None) se non interseca.
    """
    # Vettore segmento
    sx = x2 - x1
    sy = y2 - y1
    denom = dx * sy - dy * sx
    if abs(denom) < 1e-6:
        return None, None  # parallelo o coincidente
    # Risolvi il sistema:
    # p + t*d = (x1,y1) + u*(s)
    # => t*d - u*s = (x1 - px, y1 - py)
    dx1 = x1 - px
    dy1 = y1 - py
    t = (dx1 * sy - dy1 * sx) / denom
    u = (dx1 * dy - dy1 * dx) / denom
    if t >= 0 and 0 <= u <= 1:
        return t, u
    return None, None


class Robot:
    def __init__(self, x, y, radius, speed, lidar_num_rays, lidar_max_distance, environment):
        self.x = x
        self.y = y
        self.radius = radius
        self.footprint_cells = int(math.ceil(self.radius / CELL_SIDE)) - 1
        self.speed = speed
        self.LIDAR_NUM_RAYS = lidar_num_rays
        self.LIDAR_MAX_DISTANCE = lidar_max_distance
        self.environment = environment
        self.angle = random.uniform(0, 2*math.pi)
        self.grid = np.full((MAP_GRID_SIZE, MAP_GRID_SIZE), 0, dtype=np.int16)
        self.epsilon = 1e-5
        # Calcola offset per mappa interna: posiziona il robot al centro della griglia
        # grid_center = MAP_GRID_SIZE // 2
        # robot_cell_x = int(self.x // CELL_SIDE)
        # robot_cell_y = int(self.y // CELL_SIDE)
        # self.map_offset_x = grid_center - robot_cell_x
        # self.map_offset_y = grid_center - robot_cell_y
        # print(f'coordinate: {self.x} {self.y}')
        # print(f'start cell: {robot_cell_x} {robot_cell_y}')
        # print(f'offset: {self.map_offset_x} {self.map_offset_y}')
        # print(grid_center, robot_cell_x + self.map_offset_x, robot_cell_y + self.map_offset_y)

    def move_random(self):
        # Random small angle variation
        self.angle += random.uniform(-0.3, 0.3)
        dx = math.cos(self.angle) * self.speed
        dy = math.sin(self.angle) * self.speed
        nx, ny = self.x + dx, self.y + dy
        # Check wall collision
        if nx - self.radius < self.environment.x_start or nx + self.radius > self.environment.x_end:
            self.angle = math.pi - self.angle
            return
        if ny - self.radius < self.environment.y_start or ny + self.radius > self.environment.y_end:
            self.angle = -self.angle
            return
        self.x, self.y = nx, ny

    def sense_lidar(self):
        rays = []
        step_size = CELL_SIDE / 10
        for i in range(self.LIDAR_NUM_RAYS):
            ray_angle = self.angle + (i / self.LIDAR_NUM_RAYS) * 2 * math.pi
            dx, dy = math.cos(ray_angle), math.sin(ray_angle)
            dist, hit_x, hit_y = self.cast_ray(ray_angle)
            # determino la distanza massima da marcare
            max_range = dist if dist >= 0 else self.LIDAR_MAX_DISTANCE
            # campiono lungo la direzione per segnare liberi
            steps = int(math.ceil(max_range / step_size))
            for s in range(1, steps + 1):
                px = self.x + dx * min(s * step_size, max_range)
                py = self.y + dy * min(s * step_size, max_range)
                gx = int((px + self.epsilon) // CELL_SIDE)
                gy = int((py + self.epsilon) // CELL_SIDE)
                if 0 <= gx < MAP_GRID_SIZE and 0 <= gy < MAP_GRID_SIZE:
                    if dx < 0:
                        gx += 1
                    else:
                        gx -= 1
                    if dy < 0:
                        gy += 1
                    else:
                        gy -= 1
                    if self.grid[gy, gx] < 2:
                        self.grid[gy, gx] = 1  # free
            # se c'è un impatto, segna l'ostacolo
            if dist >= 0 and hit_x is not None and hit_y is not None:
                gx = int((hit_x + self.epsilon) // CELL_SIDE)
                gy = int((hit_y + self.epsilon) // CELL_SIDE)
                if 0 <= gx < MAP_GRID_SIZE and 0 <= gy < MAP_GRID_SIZE:
                    self.grid[gy, gx] = -1  # static obstacle
            else:
                # questo mi serve per disegnare anche i raggi del LiDAR che non colpiscono niente
                hit_x = self.x + math.cos(ray_angle) * self.LIDAR_MAX_DISTANCE
                hit_y = self.y + math.sin(ray_angle) * self.LIDAR_MAX_DISTANCE
            rays.append((hit_x, hit_y))
        return rays

    def cast_ray(self, angle):
        # Direzione del raggio
        dx = math.cos(angle)
        dy = math.sin(angle)
        nearest_t = None
        hit_x = None
        hit_y = None
        # Scorri ogni segmento dell'ambiente
        for seg in self.environment.walls:
            x1, y1, x2, y2 = seg
            t, u = ray_segment_intersection(self.x, self.y, dx, dy, x1, y1, x2, y2)
            if t is not None and t <= self.LIDAR_MAX_DISTANCE:
                if nearest_t is None or t < nearest_t:
                    nearest_t = t
        if nearest_t is not None:
            hit_x = self.x + dx * nearest_t
            hit_y = self.y + dy * nearest_t
            return nearest_t, hit_x, hit_y
        # Nessuna intersezione trovata
        return -1, None, None

    def clean(self):
        # robot position 
        robot_center_gx = int(self.x // CELL_SIDE)
        robot_center_gy = int(self.y // CELL_SIDE)
        for dx_cell in range(-self.footprint_cells, self.footprint_cells +1):
            for dy_cell in range(-self.footprint_cells, self.footprint_cells +1):
                fill_x = robot_center_gx + dx_cell
                fill_y = robot_center_gy + dy_cell
                self.grid[fill_y, fill_x] = 4

    def draw(self, surface):
        # Robot body
        pygame.draw.circle(surface, BLUE, (int(self.x), int(self.y)), self.radius, 2)
        # Robot orientation
        orientation_x = int(self.x + math.cos(self.angle) * self.radius * 1.5)
        orientation_y = int(self.y + math.sin(self.angle) * self.radius * 1.5)
        pygame.draw.line(surface, BLUE, (int(self.x), int(self.y)), (orientation_x, orientation_y), 5)

    def draw_lidar(self, surface, rays):
        for hit_x, hit_y in rays:
            if hit_x is not None and hit_y is not None:
                pygame.draw.line(surface, BLUE, (int(self.x), int(self.y)), (int(hit_x), int(hit_y)), 1)

    def draw_map(self, surface):
        # draw occupancy grid to a given surface of size ENVIRONMENT_WIDTH x ENVIRONMENT_HEIGHT
        for y in range(MAP_GRID_SIZE):
            for x in range(MAP_GRID_SIZE):
                val = self.grid[y, x]
                if val == -2:
                    color = YELLOW      # dynamics obstacol
                elif val == -1:
                    color = BLACK       # staic obstacle
                elif val == 0:
                    color = GRAY        # unknown
                elif val == 1:
                    color = WHITE       # free
                elif val == 2:
                    color = BLUE        # robot body
                elif val == 3:
                    color = SKY_BLUE    # robot base
                elif val == 4:
                    color = GREEN       # cleaned
                else:
                    color = RED         # re-clea
                
                    
                rect = pygame.Rect(x*CELL_SIDE, y*CELL_SIDE, CELL_SIDE, CELL_SIDE)
                pygame.draw.rect(surface, color, rect)
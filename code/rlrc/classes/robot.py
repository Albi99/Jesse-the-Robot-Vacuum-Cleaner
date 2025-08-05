import numpy as np
import math
import random
import math, random
from collections import deque

from ..constants.configuration import ENVIRONMENT_SIZE, MAP_GRID_SIZE, CELL_SIDE, LABELS_INT_TO_STR, LABELS_STR_TO_INT, ACTION_TO_STRING
from .environment import ray_segment_intersection
from ..utils import point_in_poly


class Robot:
    def __init__(self, radius, speed, lidar_num_rays, lidar_max_distance, environment, battery=1, delta_battery_per_step=0.001, base_position=None):
        # internal occupancy grid
        self.grid = np.full((MAP_GRID_SIZE, MAP_GRID_SIZE), 0, dtype=np.int16)
        self.radius = radius
        # footprint in cells
        self.footprint_cells = int(math.ceil(self.radius / CELL_SIDE)) - 1
        self.environment = environment
        # charging base and robot position
        self._set_base(base_position)
        self.speed = speed
        self.LIDAR_NUM_RAYS = lidar_num_rays
        self.LIDAR_MAX_DISTANCE = lidar_max_distance
        self.angle = 0  # random.uniform(0, 2*math.pi)
        self.epsilon = 1e-6

        self.step = 0
        self.battery = battery
        self.delta_battery_per_step = delta_battery_per_step
        self.next_reward = 0
        self.total_reward = 0
        self.previus_grid = self.grid.copy()

    def _set_base(self, base_position):
        if base_position is not None:
            self.x = base_position[0]
            self.y = base_position[1]
        else:
            N = MAP_GRID_SIZE
            # offset interno rispetto al muro
            cells_in = math.ceil(self.radius / CELL_SIDE) + 1

            # 1) Raccogliamo il poligono ordinato dei muri
            poly = self.environment.walls

            # 2) Creiamo la mask booleana delle celle interne
            interior = [[False]*N for _ in range(N)]
            for gy in range(N):
                for gx in range(N):
                    px = (gx + 0.5) * CELL_SIDE
                    py = (gy + 0.5) * CELL_SIDE
                    if point_in_poly(px, py, poly):
                        interior[gy][gx] = True

            # 3) BFS multi‐sorgente per calcolare la distanza (in celle) di ogni cella interna dal bordo
            dist = [[math.inf]*N for _ in range(N)]
            dq = deque()
            # mettiamo in coda tutte le celle “esterne” a dist=0
            for gy in range(N):
                for gx in range(N):
                    if not interior[gy][gx]:
                        dist[gy][gx] = 0
                        dq.append((gy, gx))
            # passi 4‐connessi
            dirs = [(1,0),(-1,0),(0,1),(0,-1)]
            while dq:
                y, x = dq.popleft()
                for dy, dx in dirs:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < N and 0 <= nx < N and dist[ny][nx] == math.inf:
                        # possiamo attraversare solo celle interne
                        if interior[ny][nx]:
                            dist[ny][nx] = dist[y][x] + 1
                            dq.append((ny, nx))

            # 4) Troviamo tutte le celle a distanza ESATTAMENTE cells_in
            candidates = [
                (gy, gx)
                for gy in range(N)
                for gx in range(N)
                if interior[gy][gx] and dist[gy][gx] == cells_in
            ]
            if not candidates:
                raise RuntimeError("Nessuna cella valida trovata per la base con raggio = %d celle" % cells_in)

            # 5) Scegliamo una cella a caso come centro della base
            gy, gx = random.choice(candidates)

            # 6) Posizioniamo il robot al centro di quella cella (in pixel)
            self.x = (gx + 0.5) * CELL_SIDE
            self.y = (gy + 0.5) * CELL_SIDE

        # 7) footprint della base
        self._footprint('base')

    def _back_in_base(self) -> bool:
        """
        Ritorna True se almeno la metà delle celle del quadrato
        di ingombro del robot (footprint) sono etichettate come 'base'.
        """
        # centro in celle
        cx = int(self.x // CELL_SIDE)
        cy = int(self.y // CELL_SIDE)
        # raggio in celle come da footprint
        n = self.footprint_cells
        # etichetta della base
        base_lbl = LABELS_STR_TO_INT['base']
        # conteggio celle totali e di quelle 'base'
        total = (2*n + 1) ** 2
        count_base = 0

        for dy in range(-n, n+1):
            for dx in range(-n, n+1):
                xg = cx + dx
                yg = cy + dy
                if 0 <= xg < MAP_GRID_SIZE and 0 <= yg < MAP_GRID_SIZE:
                    if self.grid[yg, xg] == base_lbl:
                        count_base += 1

        # almeno qualcosa %
        return count_base >= total * 0.75

    def reset(self, x=ENVIRONMENT_SIZE//2, y=ENVIRONMENT_SIZE//2):
        self.step = 0
        self.battery = 1
        self.total_reward = 0
        if x is not None: self.x = x
        if y is not None: self.y = y
        self.grid.fill(0)

    def play_step(self, action):
        self.next_reward = 0
        self.step += 1
        if not isinstance(action, str):
            action = ACTION_TO_STRING[action]
        d_collision_point, lidar_distances, rays, status = self.move(action)
        self.grid_diff()
        
        # lidar_distances, rays = self._sense_lidar()
        # status = self.status()

        done = False
        if self._back_in_base():
            done = True
            print('back un base')
        elif self.battery < self.delta_battery_per_step:
            self.next_reward -= 100
            done = True
        
        self.total_reward += self.next_reward
        # TODO: maybe add score = cleanded area / total area to clean
        return self.next_reward, done, self.total_reward, d_collision_point, lidar_distances, rays, status

    def move(self, direction):

        # direction: one of 'up','down','left','right' or angle delta
        if isinstance(direction, str):
            if direction == 'right': self.angle = 0
            elif direction == 'down': self.angle = 0.5*math.pi
            elif direction == 'left': self.angle = math.pi
            elif direction == 'up': self.angle = 1.5*math.pi
        else:
            # angle difference (float)
            self.angle += direction

        dx = math.cos(self.angle) * self.speed
        dy = math.sin(self.angle) * self.speed
        nx, ny = self.x + dx, self.y + dy

        # limiti mondo reale (pixel) da 0 a ENVIRONMENT_SIZE
        # (simulazioe dei sensori di contatto)
        if self._out_of_environment(nx, ny) or self._check_collision(nx, ny):
            d_collision_point_x, d_collision_point_y = nx // CELL_SIDE, ny // CELL_SIDE
            # print(f"Collision detected at, real coordinates: ({nx:.2f}, {ny:.2f}) ")
            # print(f"                           internal map: ({dnx}, {dny}) ")
            self.next_reward -= 5
        else:
            d_collision_point_x, d_collision_point_y = -1, -1
            # commit e pulizia
            self.x, self.y = nx, ny
            # etichetta le celle pultie
            self._footprint('clean')
        
        lidar_distances, rays = self._sense_lidar()
        status = self.status()

        self.battery -= self.delta_battery_per_step
        self.next_reward -= self.delta_battery_per_step
        if self.battery < 0.1:
            self.next_reward -= 2
        return (d_collision_point_x, d_collision_point_y), lidar_distances, rays, status

    def move_random(self):
        self.move(random.uniform(-0.3, 0.3))

    def _out_of_environment(self, nx, ny):
        return nx - self.radius < 0 or nx + self.radius > ENVIRONMENT_SIZE or ny - self.radius < 0 or ny + self.radius > ENVIRONMENT_SIZE

    def _check_collision(self, cx, cy):
        for x1, y1, x2, y2 in self.environment.walls:
            # Calcola la distanza minima dal centro al segmento
            if self._point_segment_dist(cx, cy, x1, y1, x2, y2) <= self.radius + self.epsilon:
                return True
        return False

    def _point_segment_dist(self, px, py, x1, y1, x2, y2):
        # Vettore segmento
        sx, sy = x2 - x1, y2 - y1
        # Vettore punto->segmento inizio
        vx, vy = px - x1, py - y1
        # Proiezione scalare
        seg_len2 = sx*sx + sy*sy
        if seg_len2 == 0:
            # Punto e segmento coincidenti
            return math.hypot(vx, vy)
        t = max(0.0, min(1.0, (vx*sx + vy*sy)/seg_len2))
        proj_x, proj_y = x1 + t*sx, y1 + t*sy
        return math.hypot(px-proj_x, py-proj_y)

    def _cast_ray(self, angle):
        # Direzione del raggio
        dx, dy = math.cos(angle), math.sin(angle)
        nearest = None
        # Scorri ogni segmento (muro) dell'ambiente
        for seg in self.environment.walls:
            t,u = ray_segment_intersection(self.x, self.y, dx, dy, *seg)
            if t is not None and t<= self.LIDAR_MAX_DISTANCE and (nearest is None or t<nearest):
                nearest = t
        if nearest is not None:
            return nearest, self.x+dx*nearest, self.y+dy*nearest
        return -1, None, None

    def _sense_lidar(self):
        lidar_distances = []
        rays = []
        step_size = CELL_SIDE / 10
        for i in range(self.LIDAR_NUM_RAYS):
            ray_angle = self.angle + (i / self.LIDAR_NUM_RAYS) * 2 * math.pi
            dx, dy = math.cos(ray_angle), math.sin(ray_angle)
            dist, hit_x, hit_y = self._cast_ray(ray_angle)
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
                    if self.grid[gy, gx] < LABELS_STR_TO_INT['clean']:
                        self.grid[gy, gx] = LABELS_STR_TO_INT['free']
            # se c'è un impatto, segna l'ostacolo
            if dist >= 0 and hit_x is not None and hit_y is not None:
                lidar_distances.append(dist)
                gx = int((hit_x + self.epsilon) // CELL_SIDE)
                gy = int((hit_y + self.epsilon) // CELL_SIDE)
                if 0 <= gx < MAP_GRID_SIZE and 0 <= gy < MAP_GRID_SIZE:
                    self.grid[gy, gx] = LABELS_STR_TO_INT['static obstacle']
            else:
                lidar_distances.append(-1)
                # questo mi serve per disegnare anche i raggi del LiDAR che non colpiscono niente
                hit_x = self.x + math.cos(ray_angle) * self.LIDAR_MAX_DISTANCE
                hit_y = self.y + math.sin(ray_angle) * self.LIDAR_MAX_DISTANCE
            rays.append((hit_x, hit_y))
        return lidar_distances, rays

    def _footprint(self, label):
        # robot position inside the internal map
        robot_center_gx = int(self.x//CELL_SIDE);
        robot_center_gy = int(self.y//CELL_SIDE)
        for dx in range(-self.footprint_cells, self.footprint_cells+1):
            for dy in range(-self.footprint_cells, self.footprint_cells+1):
                gx, gy = robot_center_gx+dx, robot_center_gy+dy
                if 0<=gx<MAP_GRID_SIZE and 0<=gy<MAP_GRID_SIZE:
                    if label == 'base':
                        self.grid[gy,gx] = LABELS_STR_TO_INT[label]
                    elif label == 'clean' and self.grid[gy, gx] != LABELS_STR_TO_INT['base']:
                        self.grid[gy,gx] = LABELS_STR_TO_INT[label]
    
    def grid_diff(self):
        """
        Confronta lo snapshot precedente della griglia (self.previus_grid)
        con lo stato corrente (self.grid) e ritorna la differenza di celle per le label 0 e 3.
        """
        prev = self.previus_grid
        curr = self.grid
        # Conta occorrenze per label 0 e 3
        prev_unknown = int((prev == LABELS_STR_TO_INT['unknown']).sum())
        curr_unknown = int((curr == LABELS_STR_TO_INT['unknown']).sum())
        prev_clean = int((prev == LABELS_STR_TO_INT['clean']).sum())
        curr_clean = int((curr == LABELS_STR_TO_INT['clean']).sum())
        # Calcola delta
        delta_unknow = curr_unknown - prev_unknown
        delta_clean = curr_clean - prev_clean
        # Aggiorna snapshot precedente
        self.previus_grid = curr.copy()
        self.next_reward -= delta_unknow / 10
        self.next_reward += delta_clean

    def status(self):
        unique, counts = np.unique(self.grid, return_counts=True)
        return dict(zip(unique, counts)), self.battery
    
    def grid_view(self):
        # Parametri di griglia e label
        labels = LABELS_INT_TO_STR.keys()

        # Calcola bounding square in celle
        radius_cells = self.radius / CELL_SIDE
        side = math.ceil(radius_cells * 2)
        cx = int(self.x // CELL_SIDE)
        cy = int(self.y // CELL_SIDE)
        x0 = cx - side // 2
        y0 = cy - side // 2

        # Costruisci raggi laterali (punto, direzione)
        rays = []
        for i in range(side):
            rays.append(((x0 + i, y0 - 1), (0, -1)))      # sopra
            rays.append(((x0 + i, y0 + side), (0, 1)))    # sotto
        for i in range(side):
            rays.append(((x0 - 1, y0 + i), (-1, 0)))      # sinistra
            rays.append(((x0 + side, y0 + i), (1, 0)))    # destra

        result = []
        # Per ogni raggio, calcola count e prima distanza per ogni label
        for (sx, sy), (dx, dy) in rays:
            # Inizializza dict per count e first_dist
            counts = {lbl: 0 for lbl in labels}
            first_dist = {lbl: None for lbl in labels}
            # Scorri dalla cella adiacente fino al bordo
            dist = 0
            x, y = sx, sy
            while 0 <= x < MAP_GRID_SIZE and 0 <= y < MAP_GRID_SIZE:
                lbl = self.grid[y, x]
                # Aggiorna count
                if lbl in counts:
                    counts[lbl] += 1
                    # Se prima occorrenza, imposta first_dist
                    if first_dist[lbl] is None:
                        first_dist[lbl] = dist
                # Avanza
                x += dx
                y += dy
                dist += 1
            # Costruisci lista fissa di tuple nell'ordine di 'labels'
            ray_list = []
            for lbl in labels:
                count = counts.get(lbl, 0)
                d = first_dist.get(lbl)
                if d is None:
                    d = dist  # se label non trovata, distanza al bordo
                if count == 0:
                    d = -1
                ray_list.append((count, d))
            result.append(ray_list)

            
            flat_view = [v for ray in result for (c, d) in ray for v in (c, d)]

        return flat_view

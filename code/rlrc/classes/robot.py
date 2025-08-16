import numpy as np
import math, random
from collections import deque

from ..classes.environment import Environment
from ..utils import point_in_poly, ray_segment_intersection, too_close_to_corner
from ..constants.configuration import LABELS_INT_TO_STR, LABELS_STR_TO_INT, ACTION_TO_STRING


class Robot:
    def __init__(self, maps, rotation=None):

        # robot
        self.battery = 1    # 100%
        self.delta_battery_per_step = 0.0001    # autonomia: 500 metri = 10'000 steps
        self.radius = 22.5
        self.cell_side = 5
        self.speed = self.cell_side

        # footprint in cells
        self.footprint_cells = int(math.ceil(self.radius / self.cell_side)) - 1
        self.epsilon = 1e-6
        
        # environment, internal map, training parameters
        self.reset(maps)

        # LiDAR
        self.lidar_num_rays = 36
        self.lidar_max_distance = 400


    def _set_base(self, base_position=None):
        if base_position is not None:
            self.x = base_position[0]
            self.y = base_position[1]
        else:
            # offset interno rispetto al muro
            cells_in = math.ceil(self.radius / self.cell_side) + 1

            # 1) Raccogliamo il poligono ordinato dei muri
            poly = self.environment.walls

            # 2) Creiamo la mask booleana delle celle interne
            interior = [[False]*self.w for _ in range(self.h)]
            for gy in range(self.h):
                for gx in range(self.w):
                    px = (gx + 0.5) * self.cell_side
                    py = (gy + 0.5) * self.cell_side
                    if point_in_poly(px, py, poly):
                        interior[gy][gx] = True

            # 3) BFS multi‐sorgente per calcolare la distanza (in celle) di ogni cella interna dal bordo
            dist = [[math.inf]*self.w for _ in range(self.h)]
            dq = deque()
            # mettiamo in coda tutte le celle “esterne” a dist=0
            for gy in range(self.h):
                for gx in range(self.w):
                    if not interior[gy][gx]:
                        dist[gy][gx] = 0
                        dq.append((gy, gx))
            # passi 4‐connessi
            dirs = [(1,0),(-1,0),(0,1),(0,-1)]
            while dq:
                y, x = dq.popleft()
                for dy, dx in dirs:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < self.h and 0 <= nx < self.w and dist[ny][nx] == math.inf:
                        # possiamo attraversare solo celle interne
                        if interior[ny][nx]:
                            dist[ny][nx] = dist[y][x] + 1
                            dq.append((ny, nx))

            # 4) Troviamo tutte le celle a distanza ESATTAMENTE cells_in
            ring_candidates = [
                (gy, gx)
                for gy in range(self.h)
                for gx in range(self.w)
                if interior[gy][gx] and dist[gy][gx] == cells_in
            ]
            if not ring_candidates:
                raise RuntimeError("Nessuna cella valida trovata per la base con raggio = %d celle" % cells_in)

            # raggio di sicurezza dagli angoli (in celle)
            CORNER_SAFE_CELLS = self.footprint_cells + 1
            CORNER_SAFE_PX = CORNER_SAFE_CELLS * self.cell_side
            EPS = self.epsilon

            # raccogli i vertici unici dei segmenti (muri)
            verts = []
            for (x1, y1, x2, y2) in self.environment.walls:
                verts.append((x1, y1))
                verts.append((x2, y2))
            unique_verts = []
            for vx, vy in verts:
                if all(abs(vx-ux) > EPS or abs(vy-uy) > EPS for (ux, uy) in unique_verts):
                    unique_verts.append((vx, vy))

            candidates = [(gy, gx) for (gy, gx) in ring_candidates if not too_close_to_corner(self, gx, gy, unique_verts, CORNER_SAFE_PX)]
            if not candidates:
                raise RuntimeError(
                    "Tutte le celle a distanza %d sono troppo vicine a un angolo. "
                    "Aumenta l'area o riduci CORNER_SAFE_CELLS."
                    % cells_in
                )

            # 5) Scegli una cella a caso tra le valide
            gy, gx = random.choice(candidates)

            # 6) Posizioniamo il robot al centro di quella cella (in pixel)
            self.x = (gx + 0.5) * self.cell_side
            self.y = (gy + 0.5) * self.cell_side
            
            # --- Imposta l'angolo per puntare verso l'interno ---
            # Mappatura direzioni (dx, dy) -> (id, angolo in radianti)
            dir_map = {
                (1, 0):  (0, 0.0),            # destra
                (0, 1):  (1, 0.5*math.pi),    # giù
                (-1, 0): (2, math.pi),        # sinistra
                (0, -1): (3, 1.5*math.pi)     # su
            }

            best_dir = None
            best_dist = -1

            for (dx, dy), (id_val, ang_val) in dir_map.items():
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.w and 0 <= ny < self.h and interior[ny][nx]:
                    # distanza dal muro nella nuova cella
                    if dist[ny][nx] > best_dist:
                        best_dist = dist[ny][nx]
                        best_dir = (id_val, ang_val)

            # Se trovata direzione valida, aggiorna self.angle_id e self.angle
            if best_dir is not None:
                self.angle_half_pi, self.angle = best_dir
            else:
                # fallback se qualcosa va storto
                self.angle_half_pi, self.angle = 0, 0.0

        # 7) footprint della base
        self._footprint('base')

        self.base_position = (self.x//self.cell_side, self.y//self.cell_side)


    def _percent_on_base(self) -> bool:
        """
        Ritorna l apercentuale delle celle del quadrato
        di ingombro del robot (footprint) sovrapposte con la base.
        """
        # centro in celle
        cx = int(self.x // self.cell_side)
        cy = int(self.y // self.cell_side)
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
                if 0 <= xg < self.w and 0 <= yg < self.h:
                    if self.grid[yg, xg] == base_lbl:
                        count_base += 1

        return count_base / total


    def reset(self, maps, rotation=None):

        # set random map (environment)
        # self.environment = Environment( rotate_map( random.choice(MAPS) ) )
        self.environment = Environment( random.choice(maps), k=rotation )
        self.last_dist_to_base = 0

        # internal map
        h = self.environment.h // self.cell_side
        w = self.environment.w // self.cell_side
        self.h, self.w = h, w
        self.grid = np.full((h, w), 0, dtype=np.int16)
        self.previus_grid = self.grid.copy()

        # set random base
        self._set_base()

        # for training
        self.next_reward = 0
        self.total_reward = 0
        self.step = 0
        self.collisions = 0
        self.battery = 1


    def play_step(self, action):

        done = False
        self.next_reward = 0
        self.step += 1

        if not isinstance(action, str):
            action = ACTION_TO_STRING[action]
        collision, lidar_distances, rays, labels_count = self.move(action)

        # --- Bonus "aderenza muro" (0 celle dal muro, senza collisione) ---
        if collision[0] == 0 and len(lidar_distances) > 0:
            import numpy as np
            min_d = float(min(lidar_distances))
            cell = float(self.cell_side)

            eps = 0.1 * cell     # margine di sicurezza (10% della cella)
            k   = 0.03           # intensità del bonus
            p   = 2.0            # rende il picco più pronunciato vicino al muro

            if min_d >= eps and min_d < cell:
                # normalizza in [0,1] dentro [eps, cell), poi rovescia (più vicino => più bonus)
                u = (min_d - eps) / (cell - eps)          # 0 quando sei quasi a contatto (sicuro), 1 quando sei a 1 cella
                bonus = k * (1.0 - u) ** p
                self.next_reward += bonus

        # bonus exploreation and cleaning
        self.grid_diff()

        clean_key = np.int16(LABELS_STR_TO_INT['clean'])
        if clean_key in labels_count:
            clean = labels_count[clean_key]
        else:
            clean = 0
        free = labels_count[np.int16(LABELS_STR_TO_INT['free'])]
        clean_over_free = (clean / (clean + free))

        # if step over base
        if self._percent_on_base() > 0:
            # if is too early
            if clean_over_free < .8:
                self.next_reward -= 1.0 * self._percent_on_base()
        
        # back in base
        if self._percent_on_base() > .8 and \
            (clean_over_free > 0.8 or self.battery < 0.2 ):
            # end episode
            done = True
            # if too early
            if clean_over_free < .8:
                self.next_reward -= 1.0
            # if ok
            else:
                self.next_reward += clean_over_free
        
        # nudge to base
        if clean_over_free >= .8:
            current_dist = self._dist_to_base() 
            if current_dist < self.last_dist_to_base:
                self.next_reward += 0.1 * (self.last_dist_to_base - current_dist)
            self.last_dist_to_base = current_dist
        
        # se non rientra in base
        if self.battery < self.delta_battery_per_step:
            self.next_reward -= 1.0
            done = True

        # penality for battery consume (penality step)
        self.next_reward -= (1 - self.battery) / 10.0
        
        self.total_reward += self.next_reward
        # TODO: maybe add score = cleanded area / total area to clean
        return self.next_reward, done, self.total_reward, collision, lidar_distances, rays, labels_count, self.battery


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
            self.collisions += 1
            has_collision = 1   # True

            # offset dal centro verso il bordo del robot lungo self.angle
            cos = math.cos(self.angle)
            sin = math.sin(self.angle)
            ox = cos * self.radius
            oy = sin * self.radius

            if cos == 1: ox -= 1
            if sin == 1: oy -= 1

            px = nx + ox
            py = ny + oy

            # converto in coordinate di cella
            d_collision_point_x = int(px // self.cell_side)
            d_collision_point_y = int(py // self.cell_side)

            # self.grid[d_collision_point_y, d_collision_point_x] = LABELS_STR_TO_INT['static obstacle']
            self.next_reward -= 0.1
        else:
            has_collision = 0   # False
            d_collision_point_x, d_collision_point_y = 0, 0
            # commit e pulizia
            self.x, self.y = nx, ny
            # etichetta le celle pultie
            self._footprint('clean')
        
        lidar_distances, rays = self._sense_lidar()
        labels_count = self.labels_count()

        self.battery -= self.delta_battery_per_step
        return (has_collision, d_collision_point_x, d_collision_point_y), lidar_distances, rays, labels_count


    def move_random(self):
        self.move(random.uniform(-0.3, 0.3))

    
    def leave_base(self):
        for _ in range(int(self.radius*2//self.cell_side)):
            collision, lidar_distances, rays, labels_count = self.move(0)
        
        self.next_reward = 0
        reward = score = 0
        done = False

        return reward, done, score, collision, lidar_distances, rays, labels_count, self.battery


    def _out_of_environment(self, nx, ny):
        return nx - self.radius < 0 or nx + self.radius > self.environment.w or ny - self.radius < 0 or ny + self.radius > self.environment.h


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
            if t is not None and t<= self.lidar_max_distance and (nearest is None or t<nearest):
                nearest = t
        if nearest is not None:
            return nearest, self.x+dx*nearest, self.y+dy*nearest
        return -1, None, None


    def _sense_lidar(self):
        lidar_distances = []
        rays = []
        step_size = self.cell_side / 10
        for i in range(self.lidar_num_rays):
            ray_angle = self.angle + (i / self.lidar_num_rays) * 2 * math.pi
            dx, dy = math.cos(ray_angle), math.sin(ray_angle)
            dist, hit_x, hit_y = self._cast_ray(ray_angle)
            # determino la distanza massima da marcare
            max_range = dist if dist >= 0 else self.lidar_max_distance
            # campiono lungo la direzione per segnare liberi
            steps = int(math.ceil(max_range / step_size))
            for s in range(1, steps + 1):
                px = self.x + dx * min(s * step_size, max_range)
                py = self.y + dy * min(s * step_size, max_range)
                gx = int((px + self.epsilon) // self.cell_side)
                gy = int((py + self.epsilon) // self.cell_side)
                if 0 <= gx < self.w and 0 <= gy < self.h:
                    if dx < 0:
                        gx += 1
                    else:
                        gx -= 1
                    if dy < 0:
                        gy += 1
                    else:
                        gy -= 1
                    if self.grid[gy, gx] < LABELS_STR_TO_INT['clean'] \
                        and self.grid[gy, gx] != LABELS_STR_TO_INT['static obstacle']:
                        self.grid[gy, gx] = LABELS_STR_TO_INT['free']
            # se c'è un impatto, segna l'ostacolo
            if dist >= 0 and hit_x is not None and hit_y is not None:
                lidar_distances.append(dist)
                gx = int((hit_x + self.epsilon) // self.cell_side)
                gy = int((hit_y + self.epsilon) // self.cell_side)
                if 0 <= gx < self.w and 0 <= gy < self.h:
                    if math.sin(self.angle) == 1: gy -= 1
                    if math.cos(self.angle) == 1: gx -= 1
                    self.grid[gy, gx] = LABELS_STR_TO_INT['static obstacle']
            else:
                # se non c'è impatto segno self.lidar_max_distance
                lidar_distances.append(self.lidar_max_distance)
                # questo mi serve per disegnare anche i raggi del LiDAR che non colpiscono niente
                hit_x = self.x + math.cos(ray_angle) * self.lidar_max_distance
                hit_y = self.y + math.sin(ray_angle) * self.lidar_max_distance
            rays.append((hit_x, hit_y))
        return lidar_distances, rays


    def _footprint(self, label):
        # robot position inside the internal map
        robot_center_gx = int(self.x//self.cell_side);
        robot_center_gy = int(self.y//self.cell_side)
        for dx in range(-self.footprint_cells, self.footprint_cells+1):
            for dy in range(-self.footprint_cells, self.footprint_cells+1):
                gx, gy = robot_center_gx+dx, robot_center_gy+dy
                if 0<=gx<self.w and 0<=gy<self.h:
                    if label == 'base':
                        self.grid[gy,gx] = LABELS_STR_TO_INT[label]
                    elif label == 'clean' \
                        and self.grid[gy, gx] != LABELS_STR_TO_INT['base'] \
                        and self.grid[gy, gx] != LABELS_STR_TO_INT['static obstacle']:
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

        # bonus exploration
        self.next_reward += - delta_unknow / 500

        # bonus cleaning
        self.next_reward += delta_clean / 100

        # bonus cleaning edge
        CLEAN = LABELS_STR_TO_INT['clean']
        OBST  = LABELS_STR_TO_INT['static obstacle']

        curr_clean_mask = (curr == CLEAN)
        prev_clean_mask = (prev == CLEAN)
        new_clean = curr_clean_mask & (~prev_clean_mask)

        obst = (curr == OBST)

        # 4-neighborhood of obstacles (no convoluzioni, solo shift)
        adj_obst = np.zeros_like(obst, dtype=bool)
        adj_obst[:-1, :] |= obst[1:, :]   # up
        adj_obst[1:,  :] |= obst[:-1, :]  # down
        adj_obst[:, :-1] |= obst[:, 1:]   # left
        adj_obst[:, 1: ] |= obst[:, :-1]  # right

        edge_new_clean = new_clean & adj_obst
        edge_count = int(edge_new_clean.sum())

        self.next_reward += edge_count / 50.0
 

    def labels_count(self):
        unique, counts = np.unique(self.grid, return_counts=True)
        lc = {int(k): int(v) for k, v in zip(unique, counts)}
        for label_int in LABELS_INT_TO_STR.keys():
            if label_int not in lc:
                lc[label_int] = 0
        return lc
    

    def grid_view(self):
        # Parametri di griglia e label
        labels = LABELS_INT_TO_STR.keys()

        # Calcola bounding square in celle
        radius_cells = self.radius / self.cell_side
        side = math.ceil(radius_cells * 2)
        cx = int(self.x // self.cell_side)
        cy = int(self.y // self.cell_side)
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
            while 0 <= x < self.w and 0 <= y < self.h:
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
                    d = 0
                ray_list.append((count, d))
            result.append(ray_list)

            flat_view = [v for ray in result for (c, d) in ray for v in (c, d)]

        return flat_view
    

    def _extract_submatrix_flat(self, offset=10):
        """
        Ritorna una patch 31x31 INT con padding -2 (out of map), SENZA rimuovere il 9x9 centrale.
        """
        import numpy as np, math
        cx = int(self.x // self.cell_side)
        cy = int(self.y // self.cell_side)
        cell_radius = int(math.ceil(self.radius / self.cell_side))
        half_ext = cell_radius + offset   # con radius=22.5, cell_side=5 => ~9; 9+10=19 -> 2*19+1=39 (adatta se vuoi 31)
        # Forziamo 31 fisso: half_ext = 15
        half_ext = 15
        size = 2 * half_ext + 1           # 31

        # padding vettoriale
        pad_y = (max(0, half_ext - cy), max(0, cy + half_ext + 1 - self.h))
        pad_x = (max(0, half_ext - cx), max(0, cx + half_ext + 1 - self.w))
        sub = self.grid[max(0, cy - half_ext): min(self.h, cy + half_ext + 1),
                        max(0, cx - half_ext): min(self.w, cx + half_ext + 1)]
        sub = np.pad(sub, (pad_y, pad_x), mode='constant', constant_values=-2)
        # garantisci shape 31x31
        if sub.shape != (31, 31):
            sub = sub[:31, :31]
        return sub  # np.int16 (31,31)


    def _dist_to_base(self):
        return math.sqrt((self.base_position[0] - self.x//self.cell_side)**2 + (self.base_position[1] - self.y//self.cell_side)**2)

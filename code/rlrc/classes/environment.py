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


class Environment:
    def __init__(self, walls):
        # Walls defined as line segments: (x1,y1)-(x2,y2)
        self.walls = walls

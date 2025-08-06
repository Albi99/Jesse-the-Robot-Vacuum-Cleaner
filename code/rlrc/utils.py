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

def point_in_poly(x, y, poly):
    """
    Ray‐casting algorithm: ritorna True se (x,y) è dentro il poligono poly = [(x1,y1),...].
    """
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0, x1, y1 = poly[i]
        # controlla se il raggio orizzontale verso +∞ interseca l’arco (y0,y1)
        if ((y0 > y) != (y1 > y)) and (x < (x1-x0)*(y-y0)/(y1-y0) + x0):
            inside = not inside
    return inside


################################


import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    # plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


######## Functions for polygons ########


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


def too_close_to_corner(robot, gx, gy, unique_verts, CORNER_SAFE_PX):
    px = (gx + 0.5) * robot.cell_side
    py = (gy + 0.5) * robot.cell_side
    for (vx, vy) in unique_verts:
        dx = px - vx
        dy = py - vy
        if dx*dx + dy*dy <= CORNER_SAFE_PX * CORNER_SAFE_PX:
            return True
    return False# utils di rotazione per le mappe (0, 90, 180, 270 gradi CCW)




######## Functions for plotting ########


import matplotlib.pyplot as plt
from IPython import display


def setup_plot():
    plt.ion()

    # creo la figura e gli assi UNA SOLA VOLTA
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

    # Disabilita la chiusura della finestra con doppio click o X
    try:
        fig.canvas.manager.window.protocol("WM_DELETE_WINDOW", lambda: None)
    except Exception:
        pass  # su backend non-Tk questa parte può non servire

    return fig, ax1, ax2

def plot_training(fig, ax1, ax2, scores, mean_scores, collisions, battery_s, clean_over_free_s):
    display.clear_output(wait=True)

    # --- Primo grafico ---
    ax1.cla()  # clear axis
    ax1.set_title('Return & Collisions')
    ax1.set_xlabel('Number of episodes')
    # ax1.set_ylabel('Return (total reward)')
    ax1.plot(range(1, len(scores) + 1), scores, label='Score')
    ax1.plot(range(1, len(mean_scores) + 1), mean_scores, label='Mean Score')
    ax1.plot(range(1, len(collisions) + 1), collisions, label='Collisions')
    ax1.legend()
    if scores:
        ax1.text(len(scores)-1, scores[-1], f"{scores[-1]:.2f}")
    if mean_scores:
        ax1.text(len(mean_scores)-1, mean_scores[-1], f"{mean_scores[-1]:.2f}")
    if collisions:
        ax1.text(len(collisions)-1, collisions[-1], collisions[-1])

    # --- Secondo grafico ---
    ax2.cla()
    ax2.set_title('Battery Level & Cleaned Area')
    ax2.set_xlabel('Number of episodes')
    ax2.set_ylabel('%')
    ax2.plot(range(1, len(battery_s) + 1), battery_s, label='Battery Level')
    ax2.plot(range(1, len(clean_over_free_s) + 1), clean_over_free_s, label='Cleaned Area')
    ax2.set_ylim(0, 110)
    ax2.legend()
    if battery_s:
        ax2.text(len(battery_s)-1, battery_s[-1], f"{battery_s[-1]:.2f}")
    if clean_over_free_s:
        ax2.text(len(clean_over_free_s)-1, clean_over_free_s[-1], f"{clean_over_free_s[-1]:.2f}")

    # --- Mostra ---
    plt.tight_layout()
    display.display(fig)
    plt.pause(0.1)




######## Functions for scaling ########


def min_max_scaling(val, min, max):
    return (val - min) / (max- min)




######## ... ########


# ...

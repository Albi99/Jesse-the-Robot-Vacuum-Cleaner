
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



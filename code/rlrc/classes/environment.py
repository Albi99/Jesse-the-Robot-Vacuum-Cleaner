import random


class Environment:

    def __init__(self, map, k: int | None = None):
        self.rotate_map(map, k)


    def rotate_map(self, map, k: int | None = None):
        """
        Ruota la mappa di k * 90° in senso antiorario (CCW) attorno all'origine (0,0),
        ricampionando le coordinate nel nuovo frame con (0,0) in alto a sinistra.
        Se k è None, viene scelto random tra {0,1,2,3}.
        
        map: ((h, w), walls) dove walls è lista di segmenti (x1,y1,x2,y2)
        ritorna:  ((h', w'), walls') con segmenti ruotati
        """
        (h, w), walls = map
        if k is None:
            k = random.choice([0, 1, 2, 3])
        k = int(k) % 4

        # dimensioni dopo la rotazione (per 90° e 270° si scambiano)
        if k % 2 == 0:
            h2, w2 = h, w
        else:
            h2, w2 = w, h

        # trasformazioni per coordinate continue con origine (0,0) in alto-sx
        # CCW: 0°, 90°, 180°, 270°
        if k == 0:
            def T(x, y): return (x, y)
        elif k == 1:
            # 90° CCW: (x', y') = (y, w - x)
            def T(x, y): return (y, w - x)
        elif k == 2:
            # 180°: (x', y') = (w - x, h - y)
            def T(x, y): return (w - x, h - y)
        else:  # k == 3
            # 270° CCW: (x', y') = (h - y, x)
            def T(x, y): return (h - y, x)

        walls2 = []
        for (x1, y1, x2, y2) in walls:
            nx1, ny1 = T(float(x1), float(y1))
            nx2, ny2 = T(float(x2), float(y2))
            # cast a int se vuoi mantenere coordinate intere
            walls2.append((int(nx1), int(ny1), int(nx2), int(ny2)))

        self.h, self.w = (h2, w2)
        # Walls defined as line segments: (x1,y1)-(x2,y2)
        self.walls = walls2

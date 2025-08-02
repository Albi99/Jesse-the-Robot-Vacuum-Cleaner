# Walls defined as line segments: (x1,y1)-(x2,y2)



# I quattro muri perimetrali, definiti come segmenti (x1,y1,x2,y2):
from .configuration import MAP_GRID_SIZE
x_start = MAP_GRID_SIZE // 2 - 300
x_end = MAP_GRID_SIZE // 2 + 300
y_start = MAP_GRID_SIZE // 2 - 200
y_end = MAP_GRID_SIZE // 2 + 200

MAP_1 = [
            (x_start, y_start, x_end, y_start),               # muro superiore
            (x_start, y_start, x_start, y_end),         # muro a destra
            (x_start, y_end, x_end, y_end),     # muro inferiore
            (x_end, y_start, x_end, y_end)               # muro a sinistra
        ]



MAP_2 = [
    (50, 200, 650, 200),
    (50, 450, 300, 450),
    (400, 450, 650, 450),
    (50, 650, 450 , 650),

    (50, 200, 50, 650),
    (450, 450, 450, 650),
    (650, 200, 650, 450)
]


MAP_3 = [
    (50, 150, 650, 150),
    (50, 400, 300, 400),
    (50, 450, 300, 450),
    (400, 400, 650, 400),
    (50, 450, 300 , 450),
    (400, 450, 450, 450),
    (50, 650, 450, 650),

    (50, 150, 50, 400),
    (50, 450, 50, 650),
    (300, 400, 300, 450),
    (400, 400, 400, 450),
    (450, 450, 450, 650),
    (650, 150, 650, 400)
]

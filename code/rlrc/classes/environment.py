class Environment:
    def __init__(self, map):

        self.h, self.w = map[0]

        # Walls defined as line segments: (x1,y1)-(x2,y2)
        self.walls = map[1]

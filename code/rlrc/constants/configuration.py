
# Robot
ROBOT_RADIUS = 22.5
ROBOT_SPEED = 5.0  # pixels per frame

# LiDAR
LIDAR_NUM_RAYS = 36
LIDAR_MAX_DISTANCE = 400    

# Internal map
ENVIRONMENT_SIZE = 700
CELL_SIDE = 5  # pixel
MAP_GRID_SIZE = ENVIRONMENT_SIZE // CELL_SIDE     # Numero di celle (per lato) nella mappa interna

LABELS_INT_TO_STR = {
    # -2 : "dynamic obstacle",
    -1 : "static obstacle",
      0: "unknown",
      1: "free",
      2: "clean",
      10: "base"
    #  4: "re-cleaned"
}

LABELS_STR_TO_INT = {
    "static obstacle": -1,
    "unknown" : 0,
    "free": 1,
    "clean": 2,
    "base": 10
}

ACTION_TO_STRING = ['right', 'down', 'left', 'up']

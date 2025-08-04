
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

LABELS = {
    # -2 : "dynamic obstacle",
    -1 : "static obstacle",
      0: "unknown",
      1: "free",
      2: "robot base",
      3: "cleaned",
    #  4: "re-cleaned"
}

ACTION_TO_STRING = ['right', 'down', 'left', 'up']

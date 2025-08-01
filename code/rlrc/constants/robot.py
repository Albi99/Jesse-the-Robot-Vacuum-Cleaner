
# Robot
ROBOT_RADIUS = 15
ROBOT_SPEED = 10.0  # pixels per frame

# LiDAR
LIDAR_NUM_RAYS = 36
LIDAR_MAX_DISTANCE = 400    

# Internal map
MAP_GRID_SIZE = 700     # Numero di celle (per lato) nella mappa interna
CELL_SIDE = 5

LABELS = {
    -2 : "dynamic obstacle",
    -1 : "static obstacle",
      0: "unknown",
      1: "free",
      2: "robot base",
      3: "cleaned",
      4: "re-cleaned"
}
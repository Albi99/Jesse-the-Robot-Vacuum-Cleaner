import pygame

from .classes.environment import Environment
from .classes.robot import Robot
from .constants.robot import MAP_GRID_SIZE, ROBOT_RADIUS, ROBOT_SPEED, LIDAR_NUM_RAYS, LIDAR_MAX_DISTANCE
from .constants.colors import WHITE, TMP_BACKGROUND


def main():
    pygame.init()
    screen = pygame.display.set_mode((MAP_GRID_SIZE*2 + 150, MAP_GRID_SIZE + 100))
    pygame.display.set_caption("Robot Vacuum Prototype")

    environment = Environment()
    # il robot parte al centro della stanza
    robot = Robot(MAP_GRID_SIZE//2, MAP_GRID_SIZE//2, ROBOT_RADIUS, ROBOT_SPEED, LIDAR_NUM_RAYS, LIDAR_MAX_DISTANCE, environment)

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Simulation step
        robot.move_random()
        robot.clean()
        rays = robot.sense_lidar()

        # Drawing
        screen.fill(TMP_BACKGROUND)
        # Left: real environment
        real_world_surface = screen.subsurface((50, 50, MAP_GRID_SIZE, MAP_GRID_SIZE))
        real_world_surface.fill(WHITE)
        environment.draw(real_world_surface)
        robot.draw_lidar(real_world_surface, rays)
        robot.draw(real_world_surface)
        # Right: internal map
        internal_map_surface = screen.subsurface((MAP_GRID_SIZE + 100, 50, MAP_GRID_SIZE, MAP_GRID_SIZE))
        internal_map_surface.fill(WHITE)
        robot.draw_map(internal_map_surface)
        robot.draw(internal_map_surface)
        # robot.draw_lidar(internal_map_surface, rays)

        pygame.display.flip()       # Update all the screen
        clock.tick(60)              # ~60 FPS

    pygame.quit()

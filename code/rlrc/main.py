import pygame

from .classes.environment import Environment
from .classes.robot import Robot
from .classes.graphics import Graphics
from .constants.maps import MAP_1, MAP_2, MAP_3, MAP_4


def main():

    environment = Environment(MAP_3)
    # il robot parte al centro della stanza
    robot = Robot(environment)
    graphics = Graphics(environment, robot)

    running = True
    while running:

        # Human
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False          # chiusura con “X”
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False      # chiusura con Esc

                elif event.key == pygame.K_UP:
                    action = 'up'
                elif event.key == pygame.K_DOWN:
                    action = 'down'
                elif event.key == pygame.K_LEFT:
                    action = 'left'
                elif event.key == pygame.K_RIGHT:
                    action = 'right'

        if action is not None:
            # rays, status = robot.move(action)
            reward, done, score, d_collision_point, lidar_distances, rays, grid_status = robot.play_step(action)
            graphics.update(rays, grid_status, score)
            # print(robot._extract_submatrix_flat())
            print(f'action: {action}, reward: {robot.next_reward}')



        # Random
        # rays, counts = robot.move_random()
        # graphics.update(rays, counts)


        # AI
        # ...


    pygame.quit()

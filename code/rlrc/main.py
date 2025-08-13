import pygame

from .classes.robot import Robot
from .classes.graphics import Graphics
from .constants.maps import MAPS_TRAIN, MAPS_TEST


def main():

    robot = Robot(MAPS_TRAIN[-1])
    graphics = Graphics(robot)

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
                
                elif event.key == pygame.K_r:
                    robot.reset()
                    graphics.reset(robot)

        if action is not None:
            reward, done, score, collision, lidar_distances, rays, labels_count, battery = robot.play_step(action)
            graphics.update(robot.environment, robot, rays, (labels_count, battery), score)
            # robot.print_state(collision, lidar_distances)

    pygame.quit()

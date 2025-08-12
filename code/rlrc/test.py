import random

from .classes.robot import Robot
from .classes.graphics import Graphics
from .utils import setup_plot, plot_training


def test_shit():
    # environment = Environment(MAP_3)
    # robot = Robot(environment)
    
    # print( list(robot.labels_count().values()) )
    # print( robot._extract_submatrix_flat() )
    # print( robot.grid_view() )
    pass


def test_canvas():

    # robot = Robot()
    # graphics = Graphics(robot)
    # # fig, ax1, ax2 = setup_plot()

    # while True:
    #     for _ in range(12):
    #         action = random.choice([0, 1, 2, 3])
    #         reward, done, score, collision, lidar_distances, rays, labels_count, battery = robot.play_step(action)
    #         graphics.update(robot.environment, robot, rays, (labels_count, battery), score)
        
    #     robot.reset()
    #     graphics.reset(robot)

    pass


def test_all_maps():

    import pygame

    from .constants.maps import MAPS_TRAIN, MAPS_TEST

    robot = Robot(MAPS_TEST[0])
    graphics = Graphics(robot)

    # train
    # for level in range(4):
    #     for map in range(8):
    #         for rotation in range(4):
    #             print(f'\nTRAIN  ==>  level: {level+1}, map: {map+1}, rotation: {rotation}')
    #             robot.reset(maps=[ MAPS_TRAIN[level][map] ], rotation=rotation)
    #             graphics.reset(robot)
    #             go = True
    #             while go:
    #                 action = random.choice([0, 1, 2, 3])
    #                 reward, done, score, collision, lidar_distances, rays, labels_count, battery = robot.play_step(action)
    #                 graphics.update(robot.environment, robot, rays, (labels_count, battery), score)
    #                 for event in pygame.event.get():
    #                     if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
    #                         go = False

    
    # test
    for level in range(4):
        for map in range(2):
            for rotation in range(4):
                print(f'\nTEST  ==>  level: {level+1}, map: {map+1}, rotation: {rotation}')
                robot.reset(maps=[ MAPS_TEST[level][map] ], rotation=rotation)
                graphics.reset(robot)
                go = True
                while go:
                    action = random.choice([0, 1, 2, 3])
                    reward, done, score, collision, lidar_distances, rays, labels_count, battery = robot.play_step(action)
                    graphics.update(robot.environment, robot, rays, (labels_count, battery), score)
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                            go = False

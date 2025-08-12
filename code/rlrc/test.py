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

    robot = Robot()
    graphics = Graphics(robot)
    # fig, ax1, ax2 = setup_plot()

    while True:
        for _ in range(12):
            action = random.choice([0, 1, 2, 3])
            reward, done, score, collision, lidar_distances, rays, labels_count, battery = robot.play_step(action)
            graphics.update(robot.environment, robot, rays, (labels_count, battery), score)
        
        robot.reset()
        graphics.reset(robot)
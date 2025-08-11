from .classes.environment import Environment
from .classes.robot import Robot
from .constants.maps import MAP_1, MAP_2, MAP_3, MAP_4


def test_shit():
    environment = Environment(MAP_3)
    robot = Robot(environment)
    
    print( list(robot.labels_count().values()) )
    print( robot._extract_submatrix_flat() )
    print( robot.grid_view() )

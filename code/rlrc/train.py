import pygame
import torch
from torch.utils.tensorboard import SummaryWriter

from .classes.environment import Environment
from .classes.robot import Robot
from .classes.graphics import Graphics
from .classes.agent import Agent
from .utils import plot
from .constants.configuration import ROBOT_RADIUS, ROBOT_SPEED, LIDAR_NUM_RAYS, LIDAR_MAX_DISTANCE, LABELS_INT_TO_STR, CELL_SIDE
from .constants.maps import MAP_1, MAP_2, MAP_3, MAP_4


def train():

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    environment = Environment(MAP_3)
    # il robot parte al centro della stanza
    robot = Robot(ROBOT_RADIUS, ROBOT_SPEED, LIDAR_NUM_RAYS, LIDAR_MAX_DISTANCE, environment)
    graphics = Graphics(environment, robot)
    agent = Agent()

    old_d_collision_point = (-1, -1)
    old_lidar_distances, _ = robot._sense_lidar()

    running = True
    while running:
        
        # human interaction
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False          # chiusura con “X”
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False      # chiusura con Esc

        # get old state
        state_old = agent.get_state(robot, old_d_collision_point, old_lidar_distances)

        # get move
        action = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score, d_collision_point, lidar_distances, rays, grid_status = robot.play_step(action)
        state_new = agent.get_state(robot, d_collision_point, lidar_distances)

        old_d_collision_point = d_collision_point
        old_lidar_distances = lidar_distances.copy()

        # update graphics
        graphics.update(rays, grid_status, score)
        print(f'action: {action}, reward: {robot.next_reward}')

        # train short memory
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # remember
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            # train long memory, plot result
            robot.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
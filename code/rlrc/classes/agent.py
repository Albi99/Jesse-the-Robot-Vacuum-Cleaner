import torch
import random
import numpy as np
from collections import deque
import pygame

from .model import Linear_QNet, QTrainer
from ..classes.environment import Environment
from ..classes.robot import Robot
from ..classes.graphics import Graphics, plot
from ..constants.configuration import ENVIRONMENT_SIZE, MAP_GRID_SIZE, ROBOT_RADIUS, ROBOT_SPEED, LIDAR_NUM_RAYS, LIDAR_MAX_DISTANCE, LABELS, CELL_SIDE
from ..constants.maps import MAP_1, MAP_2, MAP_3, MAP_4


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        n = len(LABELS)
        cells_per_robot_side = (2 * ROBOT_RADIUS // CELL_SIDE)
        input_size = int(2 + n + 4 * cells_per_robot_side * n * 2)
        hidden_layer_size = 256
        output_size = 4

        self.model = Linear_QNet(input_size, hidden_layer_size, output_size)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)


    def get_state(self, robot):
        # current orientation
        state = [robot.angle, robot.battery,]
        
        # grid status
        grid_status = [0] * len(LABELS)
        for label, count in robot.status()[0].items():
            grid_status[label] = int(count)
        state += grid_status

        # side views
        state += robot.grid_view()

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games

        move = None
        if random.randint(0, 200) < self.epsilon:
            # move = random.choice(['up', 'down', 'left', 'right']
            move = random.randint(0, 3)     # right, down, left, up
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        return move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    environment = Environment(MAP_3)
    # il robot parte al centro della stanza
    robot = Robot(ENVIRONMENT_SIZE//2, ENVIRONMENT_SIZE//2, ROBOT_RADIUS, ROBOT_SPEED, LIDAR_NUM_RAYS, LIDAR_MAX_DISTANCE, environment)
    graphics = Graphics(environment, robot)
    agent = Agent()

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
        state_old = agent.get_state(robot)

        # get move
        action = agent.get_action(state_old)

        # perform move and get new state
        reward, done, rays, grid_status = robot.play_step(action)
        state_new = agent.get_state(robot)

        # update graphics
        graphics.update(rays, grid_status)

        # train short memory
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # remember
        agent.remember(state_old, action, reward, state_new, done)

        # if done:
        #     # train long memory, plot result
        #     robot.reset()
        #     agent.n_games += 1
        #     agent.train_long_memory()

        #     if score > record:
        #         record = score
        #         agent.model.save()

        #     print('Game', agent.n_games, 'Score', score, 'Record:', record)

        #     plot_scores.append(score)
        #     total_score += score
        #     mean_score = total_score / agent.n_games
        #     plot_mean_scores.append(mean_score)
        #     plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
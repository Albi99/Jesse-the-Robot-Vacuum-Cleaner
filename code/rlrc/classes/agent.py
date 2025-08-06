import torch
import random
import numpy as np
from collections import deque

from .model import Linear_QNet, QTrainer
from ..constants.configuration import ROBOT_RADIUS, LIDAR_NUM_RAYS, LABELS_INT_TO_STR, CELL_SIDE


MAX_MEMORY = 1_000_000
BATCH_SIZE = 1_000
LEARNING_RATE = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()

        # n = len(LABELS_INT_TO_STR)
        # cells_per_robot_side = (2 * ROBOT_RADIUS // CELL_SIDE)
        # input_size = int(4 + LIDAR_NUM_RAYS + n + 4 * cells_per_robot_side * n * 2) # 405
        input_size = 1285 # 405 + 760 (+ 1200) = 1285
        output_size = 4

        self.model = Linear_QNet(input_size, output_size)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)


    def get_state(self, robot, d_collision_point, lidar_distances):
        # current orientation, battery
        state = [robot.angle, robot.battery,]

        # collision point, lidar distances
        state += list(d_collision_point) + lidar_distances

        # grid status
        counter = len(LABELS_INT_TO_STR)
        for label, count in robot.status()[0].items():
            state += [int(count)]
            counter -= 1
        for _ in range(counter):
            state += [0]

        # (sub)grid views
        state += robot._extract_submatrix_flat()

        # side views
        state += robot.grid_view()

        return np.array(state, dtype=float)


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

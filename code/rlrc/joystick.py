import pygame

from .classes.robot import Robot
from .classes.graphics import Graphics
from .constants.maps import MAPS_TRAIN, MAPS_TEST


def joystick():

    simple_map = [ MAPS_TRAIN[0][0] ]
    complex_map = [ MAPS_TRAIN[-1][4] ]
    robot = Robot(complex_map)
    graphics = Graphics(robot)

    reward, done, score, collision, lidar_distances, rays, labels_count, battery = robot.leave_base()
    graphics.update(robot.environment, robot, rays, (labels_count, battery), score)

    # streak stright reward
    MIN_REPEAT = 1
    last_actions = []
    last_reward = []
    REPEAT_REWARD = 1.0

    def streak_stright_reward():
        nonlocal last_actions, last_reward, reward

        last_actions.append(action)
        last_reward.append(reward)
        streak = len(last_actions)

        if streak >= MIN_REPEAT:
            if all(a == last_actions[0] for a in last_actions) and \
                all(r > 0 for r in last_reward):
                    reward += REPEAT_REWARD * streak
            else: 
                last_actions = []
                last_reward = []

    # penality anti-stuck position
    prev_cell = None
    same_cell_steps = 0
    same_cell_steps_malus_coeff = 0
    STUCK_STEPS = 2
    STUCK_PENALTY = 25.0

    def penality_anti_stuck_position():
        nonlocal prev_cell, same_cell_steps, same_cell_steps_malus_coeff, reward

        cell = (robot.x // robot.cell_side, robot.y // robot.cell_side)
        if prev_cell is not None and cell == prev_cell:
            same_cell_steps += 1
            if same_cell_steps >= STUCK_STEPS:
                same_cell_steps_malus_coeff +=1
                reward -= STUCK_PENALTY * same_cell_steps_malus_coeff
        else:
            same_cell_steps = 0
            same_cell_steps_malus_coeff = 0
        prev_cell = cell

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

            penality_anti_stuck_position()
            streak_stright_reward()
            robot.next_reward += reward
            score += reward

            graphics.update(robot.environment, robot, rays, (labels_count, battery), score)
            print(f'action: {action}, reward: {robot.next_reward}')
            # robot.print_state(collision, lidar_distances)

    pygame.quit()

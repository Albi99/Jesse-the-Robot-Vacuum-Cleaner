import pygame
import numpy as np

from .classes.environment import Environment
from .classes.robot import Robot
from .classes.graphics import Graphics
from .classes.agent import Agent
from .utils import setup_plot, plot_training
from .constants.configuration import LABELS_STR_TO_INT
from .constants.maps import MAP_1, MAP_2, MAP_3, MAP_4


def train():
    plot_scores, plot_mean_scores = [], []
    battery_s, clean_over_free_s = [], []
    total_score = 0
    record = 0

    environment = Environment(MAP_3)
    robot = Robot(environment)
    graphics = Graphics(environment, robot)
    fig, ax1, ax2 = setup_plot()
    agent = Agent()

    old_collision = (0, 0, 0) # No collision
    old_lidar_distances, _ = robot._sense_lidar()

    running = True
    while running:
        # human interaction
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # stato corrente
        state_old = agent.get_state(robot, old_collision, old_lidar_distances)

        # azione dalla policy
        action, logprob, value, _ = agent.get_action(state_old)

        # step ambiente
        reward, done, score, collision, lidar_distances, rays, labels_count, battery = robot.play_step(action)
        state_new = agent.get_state(robot, collision, lidar_distances)

        # aggiorna cache per prossimo ciclo
        old_collision = collision
        old_lidar_distances = lidar_distances.copy()

        # render
        graphics.update(environment, robot, rays, (labels_count, battery), score)
        # print(f'action: {action}, reward: {robot.next_reward}')
        # print(f'min: {min(state_old)}, max: {max(state_old)}')
        # if min(state_old) < 0:
        #     print('state:')
        #     for val in state_old:
        #         print(f'    {val}')

        # memorizza nel buffer PPO
        agent.store_step(state_old, action, logprob, value, reward, done)

        # aggiorna PPO se abbiamo un rollout completo o se l'episodio termina
        if agent.step_count >= agent.rollout_steps or done:
            agent.update(last_state_np=state_new)

        if done:
            # reset episodio
            robot.reset()
            agent.n_games += 1

            if score > record:
                record = score
                agent.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            clean_key = np.int16(LABELS_STR_TO_INT['clean'])
            clean = labels_count.get(clean_key, 0)
            free = labels_count[np.int16(LABELS_STR_TO_INT['free'])]
            clean_over_free = round(clean / (clean + free) * 100, 2)
            battery_s.append(battery * 100)
            clean_over_free_s.append(clean_over_free)

            plot_training(fig, ax1, ax2, plot_scores, plot_mean_scores, battery_s, clean_over_free_s)


if __name__ == '__main__':
    train()

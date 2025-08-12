import numpy as np
from collections import deque


from .classes.robot import Robot
from .classes.graphics import Graphics
from .classes.agent import Agent
from .utils import setup_plot, plot_training
from .constants.configuration import LABELS_STR_TO_INT
from .constants.maps import MAPS_TRAIN, MAPS_TEST


global plot_scores, plot_mean_scores
global battery_s, clean_over_free_s
global total_score
global record
plot_scores, plot_mean_scores = [], []
battery_s, clean_over_free_s = [], []
total_score = 0
record = 0

# for Curriculum Learning
global level                        # ultima fascia sbloccata (indice)
global max_level
global W                            # ampezza finestra mobile
global history                      # success/fail ultimi W episodi
global BIAS_LAST                    # prob. di pescare dall’ultima fascia
level = 1
max_level = 4
W = 100
history = deque(maxlen=W)
BIAS_LAST = 0.6

# global robot
# global graphics
global fig, ax1, ax2
# global agent
global training
robot = Robot(MAPS_TEST[level])
graphics = Graphics(robot)
fig, ax1, ax2 = setup_plot()
agent = Agent()
training = True


def sample_maps(maps):
    global level, BIAS_LAST

    all_maps = maps[:level]
    if level > 1 and np.random.rand() < BIAS_LAST:
        return all_maps[-1]
    # altrimenti uniforme tra tutte
    flat = [m for L in all_maps for m in L]
    return flat


def switch_to_test():
    global history, W

    if not len(history) < W:
        if np.mean(history) >= 0.80:
            test()


def next_level():
    global history, level, max_level, training

    if np.mean(history) >= 0.80:
        if level < max_level:
            filename = f'policy_level_{level}'
            agent.save(file_name=filename)
            level += 1
            print(f'LEVEL {level} UNLOCKED !!!')
        else:
            training = False
            print('LAST LEVEL ACHIEVED !!!')


def train():
    
    global plot_scores, plot_mean_scores
    global battery_s, clean_over_free_s
    global total_score, record
    global level, history
    global fig, ax1, ax2
    global training
            
    # reset episodio
    maps = sample_maps(MAPS_TRAIN)
    robot.reset(maps)
    graphics.reset(robot)

    old_collision = (0, 0, 0) # No collision
    old_lidar_distances, _ = robot._sense_lidar()

    while training:

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
        graphics.update(robot.environment, robot, rays, (labels_count, battery), score)
        # print(f'action: {action}, reward: {robot.next_reward}')

        # memorizza nel buffer PPO
        agent.store_step(state_old, action, logprob, value, reward, done)

        # aggiorna PPO se abbiamo un rollout completo o se l'episodio termina
        if agent.step_count >= agent.rollout_steps or done:
            agent.update(last_state_np=state_new)

        if done:
            # reset episodio
            maps = sample_maps(MAPS_TRAIN)
            robot.reset(maps)
            graphics.reset(robot)
            agent.n_games += 1

            if score > record:
                record = score
                agent.save(level=level)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            clean_key = np.int16(LABELS_STR_TO_INT['clean'])
            clean = labels_count.get(clean_key, 0)
            free = labels_count[np.int16(LABELS_STR_TO_INT['free'])]
            clean_over_free = round(clean / (clean + free) * 100, 2)
            battery *= 100
            battery_s.append(battery)
            clean_over_free_s.append(clean_over_free)

            plot_training(fig, ax1, ax2, plot_scores, plot_mean_scores, battery_s, clean_over_free_s)
            print(f'# episodes: {agent.n_games}, return: {score}, record (return): {record}, cleaned area: {clean_over_free}%, battery: {round(battery, 2)}, collisions: {robot.collisions}')

            success = clean_over_free >= 80 and battery >= 20 and robot.collisions == 0
            history.append(success)
            switch_to_test()


def test():

    global total_score, record
    global W

    # reset episodio
    maps = sample_maps(MAPS_TEST)
    robot.reset(maps)
    graphics.reset(robot)

    old_collision = (0, 0, 0)
    old_lidar_distances, _ = robot._sense_lidar()

    for test_index in range(W):

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
        graphics.update(robot.environment, robot, rays, (labels_count, battery), score)

        if done:
            # reset episodio
            maps = sample_maps(MAPS_TEST)
            robot.reset(maps)
            graphics.reset(robot)

            clean_key = np.int16(LABELS_STR_TO_INT['clean'])
            clean = labels_count.get(clean_key, 0)
            free = labels_count[np.int16(LABELS_STR_TO_INT['free'])]
            clean_over_free = round(clean / (clean + free) * 100, 2)
            battery *= 100

            plot_training(fig, ax1, ax2, plot_scores, plot_mean_scores, battery_s, clean_over_free_s)
            print(f'# test: {test_index}, return: {score}, record (return): {record}, cleaned area: {clean_over_free}%, battery: {round(battery, 2)}, collisions: {robot.collisions}')

            success = clean_over_free >= 80 and battery >= 20 and robot.collisions == 0
            history.append(success)
    
    next_level()



if __name__ == '__main__':
    train()

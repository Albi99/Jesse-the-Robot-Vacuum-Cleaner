import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from .model import PolicyValueNet
from ..utils import min_max_scaling



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self):
        # --- Spazio stato/azione ---
        self.input_size = 1291
        self.n_actions  = 4

        # --- Modello + optimizer ---
        self.model = PolicyValueNet(self.input_size, self.n_actions).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)

        # --- Iperparametri PPO ---
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_eps = 0.20
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5

        self.rollout_steps = 2048       # passi per aggiornamento (totali su più episodi)
        self.epochs = 10                # epoche di ottimizzazione
        self.minibatch_size = 64

        # --- Buffer on-policy ---
        self.reset_buffer()

        # --- Contatori ---
        self.n_games = 0

    # ============ Stato dal tuo robot ============
    def get_state(self, robot, collision, lidar_distances):
        state = []

        # base position (inside grid)
        state.append( ( robot.base_position[0] // robot.cells_per_side ) / robot.w )
        state.append( ( robot.base_position[1] // robot.cells_per_side ) / robot.h )

        # current position (inside grid)
        state.append( ( robot.x // robot.cells_per_side ) / robot.w )
        state.append( ( robot.y // robot.cells_per_side ) / robot.h )

        # current orientation ( angle -> sin, cos )
        theta = float(robot.angle)
        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        state.append( min_max_scaling(val=sin_th, min=-1, max=1) )
        state.append( min_max_scaling(val=cos_th, min=-1, max=1) )

        # current battery level
        # (already between 0 and 1)
        state.append( robot.battery )

        # collision point
        has_collision = collision[0]
        dx_collision = collision[1] / robot.w
        dy_collision = collision[2] / robot.h
        state.append( has_collision )
        state.append( dx_collision )
        state.append( dy_collision )
        
        # LiDAR sensor
        lidar_distances[:] = [ray / robot.lidar_max_distance for ray in lidar_distances]
        state += lidar_distances

        # labels count
        NUM_CELLS = robot.w * robot.h
        state += [min_max_scaling(val=count, min=0, max=NUM_CELLS) for count in list(robot.labels_count().values())]

        # (sub)grid view
        state += [min_max_scaling(val=cell, min=-2, max=3) for cell in robot._extract_submatrix_flat()]

        # side views
        grid_view = robot.grid_view()
        # up and down view
        for i in range(0, int(len(grid_view)/2), 2):
            count = grid_view[i]
            first_dist = grid_view[i+1]
            state.append( min_max_scaling(val=count, min=0, max=NUM_CELLS) )
            state.append( min_max_scaling(val=first_dist, min=0, max=robot.h) )
        # left and right view
        for i in range(int(len(grid_view)/2), len(grid_view), 2):
            count = grid_view[i]
            first_dist = grid_view[i+1]
            state.append( min_max_scaling(val=count, min=0, max=NUM_CELLS) )
            state.append( min_max_scaling(val=first_dist, min=0, max=robot.w) )

        return np.array(state, dtype=np.float32)

    # ============ Azione dalla policy categoriale ============
    @torch.no_grad()
    def get_action(self, state_np: np.ndarray):
        """
        Ritorna: action (int), logprob (float), value (float), logits (torch)
        """
        state = torch.tensor(state_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1, S]
        logits, value = self.model(state)  # logits: [1, A], value: [1]
        dist = Categorical(logits=logits)
        action = dist.sample()             # training: campiona
        logprob = dist.log_prob(action)    # [1]
        return int(action.item()), float(logprob.item()), float(value.item()), logits.squeeze(0)

    # ============ Buffer handling ============
    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logprobs = []
        self.values = []

        self.step_count = 0

    def store_step(self, state_np, action, logprob, value, reward, done):
        self.states.append(state_np)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.step_count += 1

    # ============ Update PPO (con bootstrap sul last_state) ============
    def update(self, last_state_np: np.ndarray):
        if self.step_count == 0:
            return

        # Tensors
        states = torch.tensor(np.vstack(self.states), dtype=torch.float32, device=DEVICE)     # [T, S]
        actions = torch.tensor(self.actions, dtype=torch.long, device=DEVICE)                 # [T]
        old_logprobs = torch.tensor(self.logprobs, dtype=torch.float32, device=DEVICE)        # [T]
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=DEVICE)              # [T]
        dones = torch.tensor(self.dones, dtype=torch.float32, device=DEVICE)                  # [T]
        values = torch.tensor(self.values, dtype=torch.float32, device=DEVICE)                # [T]

        # Bootstrap value from last state
        with torch.no_grad():
            last_state = torch.tensor(last_state_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            _, last_value = self.model(last_state)  # scalar

        # GAE-Lambda
        advantages = torch.zeros_like(rewards, device=DEVICE)
        gae = 0.0
        for t in reversed(range(self.step_count)):
            next_value = last_value if t == self.step_count - 1 else values[t+1]
            delta = rewards[t] + self.gamma * (1 - dones[t]) * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values

        # Normalizza advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Ottimizzazione per epoche/minibatch
        dataset_size = self.step_count
        idxs = np.arange(dataset_size)

        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, self.minibatch_size):
                mb_idx = idxs[start:start + self.minibatch_size]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                logits, new_values = self.model(mb_states)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_values, mb_returns)

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # Svuota buffer
        self.reset_buffer()

    # ============ Salvataggio ============
    def save(self, file_name: str = 'model.pth'):
        self.model.save(file_name)

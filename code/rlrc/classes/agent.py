# agent.py — versione PPO con 4 encoder + trunk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from .encoders import (
    PatchEncoder2D, LidarEncoderConv1D, LidarEncoderMLP,
    SideViewsEncoder, ScalarsEncoder, RobotPolicyValueNet
)
from ..constants.configuration import LABELS_STR_TO_INT


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def one_hot_patch(flat_vals, H=31, W=31):
    """
    flat_vals: lista/array di len H*W con valori in {-2, 0..4, 5}
    Mapping canali:
      0: out-of-map (-2)
      1: static obstacle (label 'static obstacle')
      2: unknown
      3: free
      4: clean
      5: base
    Ritorna: [C=6, H, W] float32 in {0,1}
    """
    arr = np.array(flat_vals, dtype=np.int16).reshape(H, W)
    C = 6
    out = np.zeros((C, H, W), dtype=np.float32)
    # mappa etichette del tuo progetto
    lbl = LABELS_STR_TO_INT
    # canale 0: out-of-map
    out[0] = (arr == -2).astype(np.float32)
    # canale 1: static obstacle
    out[1] = (arr == lbl['static obstacle']).astype(np.float32)
    # canale 2: unknown
    out[2] = (arr == lbl['unknown']).astype(np.float32)
    # canale 3: free
    out[3] = (arr == lbl['free']).astype(np.float32)
    # canale 4: clean
    out[4] = (arr == lbl['clean']).astype(np.float32)
    # canale 5: base
    out[5] = (arr == lbl['base']).astype(np.float32)
    return out  # [6,31,31]


class Agent:
    def __init__(self):
        # ------- costruiamo la rete a encoder --------
        # Patch CNN
        self.patch_enc = PatchEncoder2D(in_channels=6, out_dim=256)

        # LiDAR: scegli una
        # self.lidar_enc = LidarEncoderMLP(in_dim=36, out_dim=64)
        self.lidar_enc = LidarEncoderConv1D(n_rays=36, out_dim=64)

        # SideViews: dimensione in ingresso = 40 * side
        # (la calcoleremo al primo get_state e poi ricreeremo la head se diverso)
        self.side_in_dim = None
        self.side_enc = None  # inizializzato lazy nel primo forward

        # Scalars: scegli cosa infilarci (qui 16 come nel tuo encoders.py)
        self.scal_enc = ScalarsEncoder(in_dim=16, out_dim=32)

        # Model: lo inizializziamo appena sappiamo side_in_dim
        self.model = None

        # ------- optimizer/iperparametri PPO --------
        self.optimizer = None
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_eps = 0.20            # was 0.20 and 0.15 and 0.10
        self.ent_coef = 0.03            # was 0.01 and 0.02 and 0.03
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5

        self.rollout_steps = 1024       # was 2048
        self.epochs = 5                 # was 10
        self.minibatch_size = 128       # was 64

        # buffer on-policy
        self.reset_buffer()
        self.n_games = 0

    # --------- costruttore rete completo (lazy) ----------
    def _ensure_model(self, side_in_dim):
        if (self.model is None) or (self.side_in_dim != side_in_dim):
            self.side_in_dim = side_in_dim
            self.side_enc = SideViewsEncoder(in_dim=side_in_dim, out_dim=32)
            self.model = RobotPolicyValueNet(
                patch_enc=self.patch_enc,
                lidar_enc=self.lidar_enc,
                side_enc=self.side_enc,
                scal_enc=self.scal_enc,
                trunk_dim=256,
                n_actions=4
            ).to(DEVICE)
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)  # was 3e-4 and 1e-4 and 1e-5

    # ============ Stato dal robot: produce i 4 blocchi ============
    def get_state(self, robot, collision, lidar_distances):
        """
        Ritorna un dict:
          {
            'patch':   np.float32 [6,31,31],
            'lidar':   np.float32 [36] in [0,1],
            'side':    np.float32 [F]  (F=40*side) normalizzato,
            'scalars': np.float32 [16] (scala [0,1] o z-score)
          }
        """
        H, W = robot.h, robot.w
        # ---- scalars ----
        # base pos (2), robot pos (2) normalizzate
        bx = (robot.base_position[0] // robot.cell_side) / W
        by = (robot.base_position[1] // robot.cell_side) / H
        rx = (robot.x // robot.cell_side) / W
        ry = (robot.y // robot.cell_side) / H

        # orientamento -> sin/cos
        th = float(robot.angle)
        sin_th, cos_th = np.sin(th), np.cos(th)

        # batteria [0,1]
        battery = float(robot.battery)

        # collision flag + coords norm
        has_col, cx_raw, cy_raw = collision
        has_col = float(has_col)
        cx = (cx_raw / W) if has_col else 0.0
        cy = (cy_raw / H) if has_col else 0.0

        # label counts (6 canali) normalizzate per #celle
        counts_dict = robot.labels_count()   # {int:count}
        NUM_CELLS = float(W * H)
        # L’ordine deve essere coerente con canali/etichette che usi:
        counts = []
        for k in LABELS_STR_TO_INT.values():
            # ATT: se values() non è ordinato, meglio iterare sulle chiavi in un ordine fissato.
            # Qui assumiamo l'ordine naturale delle etichette nel tuo dict di costanti.
            pass
        # Per evitare dipendenze sull'ordine del dict, ricostruiamo così:
        label_names = ['out of map','static obstacle','unknown','free','clean','base']
        # 'out of map' non esiste in grid_count: aggiungilo 0
        counts.append(0.0)  # out-of-map
        lbl = LABELS_STR_TO_INT
        counts.append(counts_dict.get(int(lbl['static obstacle']), 0) / NUM_CELLS)
        counts.append(counts_dict.get(int(lbl['unknown']), 0) / NUM_CELLS)
        counts.append(counts_dict.get(int(lbl['free']), 0) / NUM_CELLS)
        counts.append(counts_dict.get(int(lbl['clean']), 0) / NUM_CELLS)
        counts.append(counts_dict.get(int(lbl['base']), 0) / NUM_CELLS)

        # compone il vettore scalari (16)
        scalars = np.array([
            bx, by, rx, ry,
            sin_th, cos_th, battery,
            has_col, cx, cy,
            *counts  # 6 numeri
        ], dtype=np.float32)
        assert scalars.shape[0] == 16, f"Scalars len={scalars.shape[0]} (atteso 16)"

        # ---- lidar [36] in [0,1] ----
        lidar = np.asarray(lidar_distances, dtype=np.float32) / float(robot.lidar_max_distance)
        lidar = np.clip(lidar, 0.0, 1.0)

        # ---- patch one-hot [6,31,31] ----
        sub_flat = robot._extract_submatrix_flat()  # len == 31*31 (senza 9x9? -> nel tuo codice è già flatten senza il 9x9 centrale; se così, sostituisci con una funzione che NON rimuove il centro)
        # ATTENZIONE: la tua _extract_submatrix_flat rimuove il 9x9 centrale.
        # Per la CNN serve il patch completo 31x31. Se la tua funzione rimuove,
        # crea una variante che NON lo rimuove. Per ora assumo che restituisca 31*31.
        patch = one_hot_patch(sub_flat, H=31, W=31)  # [6,31,31]

        # ---- side views ----
        side_raw = np.asarray(robot.grid_view(), dtype=np.float32)
        # Norm conteggi/distanze per max lato
        side_len_half = side_raw.shape[0] // 2  # metà (up/down) vs (left/right)
        max_side_h = float(H)
        max_side_w = float(W)
        side = side_raw.copy()
        # up+down tratto come verticale (limite H)
        for i in range(0, side_len_half, 2):
            side[i]   = side[i]   / (max_side_h)  # count
            side[i+1] = side[i+1] / (max_side_h)  # first distance
        # left+right orizzontale (limite W)
        for i in range(side_len_half, side_raw.shape[0], 2):
            side[i]   = side[i]   / (max_side_w)
            side[i+1] = side[i+1] / (max_side_w)
        side = np.clip(side, 0.0, 1.0)

        # inizializza/reinizializza il modello se cambia side_in_dim
        self._ensure_model(side_in_dim=side.shape[0])
        
        # DEBUG
        if self.n_games < 1 and self.step_count < 5:
            print(patch.shape, lidar.shape, side.shape, scalars.shape)

        return {
            'patch':   patch.astype(np.float32),     # [6,31,31]
            'lidar':   lidar.astype(np.float32),     # [36]
            'side':    side.astype(np.float32),      # [F]
            'scalars': scalars.astype(np.float32),   # [16]
        }

    # ============ Azione dalla policy categoriale ============
    @torch.no_grad()
    def get_action(self, state_dict):
        self.model.eval()   # <-- rete in modalità eval per rollout

        # numpy -> torch
        patch = torch.tensor(state_dict['patch'],   device=DEVICE).unsqueeze(0)   # [1,6,31,31]
        lidar = torch.tensor(state_dict['lidar'],   device=DEVICE).unsqueeze(0)   # [1,36]
        side  = torch.tensor(state_dict['side'],    device=DEVICE).unsqueeze(0)   # [1,F]
        scal  = torch.tensor(state_dict['scalars'], device=DEVICE).unsqueeze(0)   # [1,16]

        logits, value = self.model(patch, lidar, side, scal)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return int(action.item()), float(logprob.item()), float(value.item()), logits.squeeze(0)

    # ============ Buffer handling ============
    def reset_buffer(self):
        self.patches = []
        self.lidars = []
        self.sides = []
        self.scalars = []

        self.actions = []
        self.rewards = []
        self.dones = []
        self.logprobs = []
        self.values = []

        self.step_count = 0

    def store_step(self, state_dict, action, logprob, value, reward, done):
        self.patches.append(state_dict['patch'])
        self.lidars.append(state_dict['lidar'])
        self.sides.append(state_dict['side'])
        self.scalars.append(state_dict['scalars'])

        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.step_count += 1

    # ============ Update PPO (con bootstrap sul last_state) ============
    def update(self, last_state_dict):
        self.model.train()  # <-- rete in modalità training per aggiornamento
        self.ent_coef = max(0.005, self.ent_coef * 0.999) # decadimento entropy
        
        if self.step_count == 0:
            return

        # Stack in torch
        patches = torch.tensor(np.stack(self.patches), dtype=torch.float32, device=DEVICE)  # [T,6,31,31]
        lidars  = torch.tensor(np.stack(self.lidars),  dtype=torch.float32, device=DEVICE)  # [T,36]
        sides   = torch.tensor(np.stack(self.sides),   dtype=torch.float32, device=DEVICE)  # [T,F]
        scalars = torch.tensor(np.stack(self.scalars), dtype=torch.float32, device=DEVICE)  # [T,16]

        actions = torch.tensor(self.actions, dtype=torch.long, device=DEVICE)               # [T]
        old_logprobs = torch.tensor(self.logprobs, dtype=torch.float32, device=DEVICE)      # [T]
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=DEVICE)            # [T]
        dones = torch.tensor(self.dones, dtype=torch.float32, device=DEVICE)                # [T]
        values = torch.tensor(self.values, dtype=torch.float32, device=DEVICE)              # [T]

        # Bootstrap value from last state
        with torch.no_grad():
            last_patch = torch.tensor(last_state_dict['patch'], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            last_lidar = torch.tensor(last_state_dict['lidar'], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            last_side  = torch.tensor(last_state_dict['side'],  dtype=torch.float32, device=DEVICE).unsqueeze(0)
            last_scal  = torch.tensor(last_state_dict['scalars'], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            _, last_value = self.model(last_patch, last_lidar, last_side, last_scal)
            last_value = last_value.squeeze(0)

        # GAE-Lambda
        advantages = torch.zeros_like(rewards, device=DEVICE)
        gae = 0.0
        for t in reversed(range(self.step_count)):
            next_value = last_value if t == self.step_count - 1 else values[t+1]
            delta = rewards[t] + self.gamma * (1 - dones[t]) * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Ottimizzazione
        dataset_size = self.step_count
        idxs = np.arange(dataset_size)

        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, self.minibatch_size):
                mb_idx = idxs[start:start + self.minibatch_size]

                mb_patch = patches[mb_idx]
                mb_lidar = lidars[mb_idx]
                mb_side  = sides[mb_idx]
                mb_scal  = scalars[mb_idx]

                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                logits, new_values = self.model(mb_patch, mb_lidar, mb_side, mb_scal)
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

        self.reset_buffer()
        with torch.no_grad():
            log_ratio = new_logprobs - mb_old_logprobs
            approx_kl = (-log_ratio).mean().clamp_min(0.0) 
            if approx_kl < 0.005: message = 'TOO LOW'
            elif approx_kl > 0.02: message = 'A BIT HIGH'
            elif approx_kl > 0.05: message = 'TOO HIGH'
            else: message = 'OK'
            print(f'approx KL {message} {approx_kl}')


    # ============ Salvataggio ============
    def save(self, file_name: str = 'model.pth', level=None):
        # riuso il tuo metodo in model.py se vuoi, qui semplice torch.save
        torch.save(self.model.state_dict(), f'./model/{file_name}')

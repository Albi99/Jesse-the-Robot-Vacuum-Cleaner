import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 1) PATCH ENCODER 2D (CNN)
# ----------------------------
class PatchEncoder2D(nn.Module):
    """
    Input:  patch one-hot di etichette [B, C, H, W], es. C=5, H=W=31
    Output: vettore feature [B, out_dim]
    """
    def __init__(self, in_channels=6, out_dim=256):
        # in_channel il the numbero of labels (out of map, static obstacle, unknown, free, clean, base)
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2),  # 31 -> 31
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),           # 31 -> 31
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                                 # 31 -> 15
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),           # 15 -> 15
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=3),                                       # 15 -> 5
            nn.Flatten(),                                                    # 96*5*5 = 2400
            nn.Linear(96*5*5, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # x: [B,C,31,31]
        return self.net(x) # [B,out_dim]


# ----------------------------
# 2) LiDAR ENCODER (due varianti)
#    - MLP semplice
#    - Conv1D per pattern locali angolari
# ----------------------------
class LidarEncoderMLP(nn.Module):
    """
    Input:  vettore distanze LiDAR [B, N]
    Output: vettore feature [B, out_dim]
    """
    def __init__(self, in_dim=36, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # x: [B,N]
        return self.net(x) # [B,out_dim]


class LidarEncoderConv1D(nn.Module):
    """
    Input:  distanze LiDAR [B, N] -> reshape a [B, 1, N]
    Output: feature [B, out_dim]
    Conv1D coglie pattern locali (ostacolo a ore 3, ecc.)
    """
    def __init__(self, n_rays=36, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),   # 36 -> 18
            nn.Conv1d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # -> [B, 48, 1]
            nn.Flatten(),             # -> [B, 48]
        )
        self.proj = nn.Sequential(
            nn.Linear(48, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):        # x: [B,N]
        x = x.unsqueeze(1)       # -> [B,1,N]
        x = self.conv(x)         # -> [B,48]
        return self.proj(x)      # -> [B,out_dim]


# ----------------------------
# 3) SIDE-VIEWS ENCODER (MLP)
# ----------------------------
class SideViewsEncoder(nn.Module):
    """
    Input: side-views [B, F] con F = 4*side*5*2 (di solito side≈9 => F≈360)
    Output: feature [B, out_dim]
    """
    def __init__(self, in_dim=40, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # x: [B,F]
        return self.net(x) # [B,out_dim]


# ----------------------------
# 4) SCALARS ENCODER (MLP)
# ----------------------------
class ScalarsEncoder(nn.Module):
    """
    Input:  vettore scalare eterogeneo [B, S]
            (batteria, contatori globali, posizioni, orientamento, collision coords, ecc.)
    Output: feature [B, out_dim]
    """
    def __init__(self, in_dim=16, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # x: [B,S]
        return self.net(x) # [B,out_dim]


# ----------------------------
# 5) FUSIONE + TESTE POLICY / VALUE
# ----------------------------
class RobotPolicyValueNet(nn.Module):
    """
    Combina i 4 encoder, concatena le feature e produce:
      - policy logits (Categorical) per 4 azioni
      - valore V(s)
    Supporta opzionalmente l'action masking (logits = -inf per azioni invalide).
    """
    def __init__(
        self,
        patch_enc: PatchEncoder2D,
        lidar_enc: nn.Module,         # scegli MLP o Conv1D
        side_enc: SideViewsEncoder,
        scal_enc: ScalarsEncoder,
        trunk_dim=256,
        n_actions=4,
    ):
        super().__init__()
        self.patch_enc = patch_enc
        self.lidar_enc = lidar_enc
        self.side_enc  = side_enc
        self.scal_enc  = scal_enc

        fused_dim = (patch_enc.net[-2].out_features  # out_dim del patch encoder
                     + lidar_enc.net[-2].out_features if isinstance(lidar_enc, LidarEncoderMLP)
                     else 0)  # placeholder, correggiamo sotto

        # calcolo generale robusto dell'output di ciascun encoder
        self._patch_out = patch_enc.net[-2].out_features
        # lidar
        if isinstance(lidar_enc, LidarEncoderMLP):
            self._lidar_out = lidar_enc.net[-2].out_features
        elif isinstance(lidar_enc, LidarEncoderConv1D):
            self._lidar_out = lidar_enc.proj[0].out_features
        else:
            raise ValueError("lidar_enc deve essere LidarEncoderMLP o LidarEncoderConv1D")

        self._side_out  = side_enc.net[-2].out_features
        self._scal_out  = scal_enc.net[-2].out_features

        fused_dim = self._patch_out + self._lidar_out + self._side_out + self._scal_out

        # trunk comune
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, trunk_dim),
            nn.ReLU(inplace=True),
            nn.Linear(trunk_dim, trunk_dim),
            nn.ReLU(inplace=True),
        )

        # heads
        self.policy_head = nn.Linear(trunk_dim, n_actions)  # logits (softmax fuori)
        self.value_head  = nn.Linear(trunk_dim, 1)

    def forward(self, patch, lidar, side, scalars, invalid_action_mask=None):
        """
        patch:   [B,C,H,W] (one-hot)
        lidar:   [B,N]
        side:    [B,F]  (o [B,4,5,2] -> flatten prima di passare)
        scalars: [B,S]
        invalid_action_mask: [B, n_actions] con 1 per azione invalida (opzionale)

        Ritorna: logits [B, n_actions], value [B,1]
        """
        f_patch = self.patch_enc(patch)
        f_lidar = self.lidar_enc(lidar)
        f_side  = self.side_enc(side)
        f_scal  = self.scal_enc(scalars)

        fused = torch.cat([f_patch, f_lidar, f_side, f_scal], dim=1)
        h = self.trunk(fused)

        logits = self.policy_head(h)  # [B, A]
        if invalid_action_mask is not None:
            # Maschera azioni invalide settando -inf sui logits corrispondenti
            # Assumi mask ∈ {0,1}, 1 = invalida
            logits = logits.masked_fill(invalid_action_mask.bool(), float('-inf'))

        value = self.value_head(h).squeeze(-1)    # [B, 1]
        return logits, value


# ----------------------------
# 6) ESEMPIO D'USO / TEST SHAPES
# ----------------------------
if __name__ == "__main__":
    B = 8                 # batch
    C, H, W = 6, 31, 31   # one-hot map
    N_RAYS = 36
    SIDE_F = 4*5*2        # 4 side * 9 cells per side * 5 labels * 2 values per label
    S_SCAL = 16

    patch = torch.randn(B, C, H, W)      # (metti qui la tua one-hot già normalizzata)
    lidar = torch.rand(B, N_RAYS)        # distanze normalizzate in [0,1]
    side  = torch.rand(B, SIDE_F)        # oppure torch.rand(B,4,5,2).view(B,-1)
    scal  = torch.randn(B, S_SCAL)       # scalari z-score o min-max
    inv_mask = torch.zeros(B, 4)         # esempio: nessuna azione invalida

    net = RobotPolicyValueNet(
        patch_enc=PatchEncoder2D(in_channels=C, out_dim=256),
        # Scegli uno dei due:
        # lidar_enc=LidarEncoderMLP(in_dim=N_RAYS, out_dim=64),
        lidar_enc=LidarEncoderConv1D(n_rays=N_RAYS, out_dim=64),
        side_enc=SideViewsEncoder(in_dim=SIDE_F, out_dim=32),
        scal_enc=ScalarsEncoder(in_dim=S_SCAL, out_dim=32),
        trunk_dim=256,
        n_actions=4,
    )

    logits, value = net(patch, lidar, side, scal, invalid_action_mask=inv_mask)
    print("logits:", logits.shape)  # [B,4]
    print("value :", value.shape)   # [B,1]

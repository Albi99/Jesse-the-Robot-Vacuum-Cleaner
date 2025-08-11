import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class PolicyValueNet(nn.Module):
    """
    MLP unico con due teste:
      - policy_head -> logits (Categorical) per 4 azioni
      - value_head  -> V(s)
    """
    def __init__(self, input_size: int, n_actions: int = 4):
        super().__init__()
        self.input_size = input_size
        h1, h2, h3 = 1024, 512, 256

        self.fc1 = nn.Linear(input_size, h1)
        self.ln1 = nn.LayerNorm(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.ln2 = nn.LayerNorm(h2)
        self.fc3 = nn.Linear(h2, h3)
        self.ln3 = nn.LayerNorm(h3)
        self.dropout = nn.Dropout(p=0.1)

        # Teste separate
        self.policy_head = nn.Linear(h3, n_actions)  # logits
        self.value_head  = nn.Linear(h3, 1)          # V(s)

        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x); x = self.ln1(x); x = F.relu(x); x = self.dropout(x)
        x = self.fc2(x); x = self.ln2(x); x = F.relu(x); x = self.dropout(x)
        x = self.fc3(x); x = self.ln3(x); x = F.relu(x); x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        logits = self.policy_head(h)
        value  = self.value_head(h).squeeze(-1)  # [B]
        return logits, value

    def save(self, file_name: str = 'model.pth'):
        model_folder = './model'
        os.makedirs(model_folder, exist_ok=True)
        path = os.path.join(model_folder, file_name)
        torch.save(self.state_dict(), path)

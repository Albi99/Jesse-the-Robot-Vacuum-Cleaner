import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        # Save expected input size
        self.input_size = input_size
        # Hidden layer sizes
        h1, h2, h3 = 1024, 512, 256
        # Define layers
        self.fc1 = nn.Linear(input_size, h1)
        self.ln1 = nn.LayerNorm(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.ln2 = nn.LayerNorm(h2)
        self.fc3 = nn.Linear(h2, h3)
        self.ln3 = nn.LayerNorm(h3)
        self.out = nn.Linear(h3, output_size)
        self.dropout = nn.Dropout(p=0.1)

        # Initialize weights with Xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First hidden block
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout(x)
        # Second hidden block
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout(x)
        # Third hidden block
        x = self.fc3(x)
        x = self.ln3(x)
        x = F.relu(x)
        x = self.dropout(x)
        # Output Q-values
        return self.out(x)

    def save(self, file_name: str = 'model.pth'):
        model_folder = './model'
        os.makedirs(model_folder, exist_ok=True)
        path = os.path.join(model_folder, file_name)
        torch.save(self.state_dict(), path)


class QTrainer:
    def __init__(self, 
                 model: nn.Module, 
                 lr: float, 
                 gamma: float, 
                 weight_decay: float = 1e-5
                 ):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # Adam with weight decay for slight regularization
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.max_grad_norm = 1.0  # for gradient clipping

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q-values with current state
        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                # Bellman update
                q_next = torch.max(self.model(next_state[idx]))
                q_new = reward[idx] + self.gamma * q_next

            # target[idx][torch.argmax(action[idx]).item()] = Q_new
            idx_action = action[idx].item() # so 0, 1, 2, or 3
            target[idx][idx_action] = q_new
    
        # # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # # pred.clone()
        # # preds[argmax(action)] = Q_new
        # self.optimizer.zero_grad()
        # loss = self.criterion(target, pred)
        # loss.backward()

        # self.optimizer.step()

        # Compute loss
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        # return loss.item()

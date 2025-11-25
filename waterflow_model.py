import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Affine Coupling Block (The Reversible Layer) ---
class AffineCoupling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # s (scaling) and t (translation) nets
        self.s = nn.Sequential(nn.Linear(dim // 2, 64), nn.ReLU(), nn.Linear(64, dim // 2), nn.Tanh())
        self.t = nn.Sequential(nn.Linear(dim // 2, 64), nn.ReLU(), nn.Linear(64, dim // 2))

    def forward(self, x):
        x_a, x_b = x.chunk(2, dim=-1)
        s, t = self.s(x_a), self.t(x_a)
        y_a = x_a
        y_b = x_b * torch.exp(s) + t
        y = torch.cat([y_a, y_b], dim=-1)
        log_det_J = s.sum(dim=-1) 
        return y, log_det_J

    def reverse(self, y):
        y_a, y_b = y.chunk(2, dim=-1)
        s, t = self.s(y_a), self.t(y_a)
        x_a = y_a
        x_b = (y_b - t) * torch.exp(-s)
        return torch.cat([x_a, x_b], dim=-1)

# --- 2. WaterFlow Model (Stacked Layers) ---
class WaterFlow(nn.Module):
    def __init__(self, input_dim, num_layers=16):
        super().__init__()
        assert input_dim % 2 == 0
        self.layers = nn.ModuleList([AffineCoupling(input_dim) for _ in range(num_layers)])
        self.prior = torch.distributions.Normal(torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))
        self.input_dim = input_dim

    def forward(self, x):
        log_det_J_total = 0.0
        z = x
        for layer in self.layers:
            z, log_det_J = layer(z)
            log_det_J_total += log_det_J
        return z, log_det_J_total

    def reverse(self, z):
        """Reverses the entire flow from Z back to X."""
        x = z
        for layer in reversed(self.layers):
            x = layer.reverse(x)
        return x

# --- 3. Thermodynamic Loss (F) with Shannon Entropy (S) ---

def thermodynamic_loss_shannon(model, x, beta, lambda_reg=1e-6):
    """
    Calculates the Thermodynamic Free Energy (F): F = beta * E(x) - lambda * S_reg
    where E(x) is the Negative Log-Likelihood (NLL) and S_reg is the Shannon Entropy of weights.
    """
    
    # 1. Calculate E(x) (NLL)
    z, log_det_J_total = model(x)
    log_p_z = model.prior.log_prob(z).sum(dim=-1)
    E_x = -(log_p_z + log_det_J_total).mean()
    
    # 2. Calculate S_reg (Shannon Entropy of Weights)
    S_reg = 0.0
    for param in model.parameters():
        if param.requires_grad:
            flat_param = param.abs().flatten()
            if flat_param.sum() == 0:
                continue
            p = flat_param / flat_param.sum()
            S_reg += -(p * torch.log(p + 1e-9)).sum()
            
    # 3. Calculate F
    F_loss = beta * E_x - lambda_reg * S_reg
    
    return F_loss, E_x.item(), S_reg.item()

# --- 4. Beta Annealing Schedule ---

def get_beta(step, total_steps_warmup):
    return min(1.0, step / total_steps_warmup)

# --- 5. Training Setup ---

if __name__ == '__main__':
    # Load MNIST Data (De-quantized and Flattened)
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: (x * 255. + torch.rand_like(x)) / 256., 
        transforms.Lambda(lambda x: x.flatten()) 
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Hyperparameters
    INPUT_DIM = 784
    NUM_LAYERS = 16 
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10 
    
    # Thermodynamic Setup
    TOTAL_STEPS = NUM_EPOCHS * len(train_loader)
    WARMUP_STEPS = int(0.5 * TOTAL_STEPS) 
    LAMBDA_REG = 1e-6 

    # Initialization
    model = WaterFlow(INPUT_DIM, NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Starting WaterFlow (Thermodynamic Training) ---")
    step_count = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            beta = get_beta(step_count, WARMUP_STEPS)
            
            optimizer.zero_grad()
            F_loss, E_x, S_reg = thermodynamic_loss_shannon(model, data, beta, LAMBDA_REG)
            
            F_loss.backward()
            optimizer.step()
            
            if step_count % 500 == 0:
                BPD = E_x * (np.log2(np.e) / INPUT_DIM) 
                print(f"E: {epoch:02d} | Step: {step_count:05d} | Beta: {beta:.3f} | "
                      f"F Loss: {F_loss.item():.4f} | BPD: {BPD:.4f} | S (Shannon): {S_reg:.2f}")

            step_count += 1
    
    print("\nTraining complete. The model has reached thermodynamic equilibrium.")

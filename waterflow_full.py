import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt
import umap
import matplotlib.animation as animation
from tqdm import tqdm
from IPython.display import HTML 

# --- DEVICE SETUP ---
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL DEFINITIONS ---

class AffineCoupling(nn.Module):
    """The core reversible layer (Change of variable)"""
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
        """Perfect Inversion (Reversibility)"""
        y_a, y_b = y.chunk(2, dim=-1)
        s, t = self.s(y_a), self.t(y_a)
        x_a = y_a
        x_b = (y_b - t) * torch.exp(-s)
        return torch.cat([x_a, x_b], dim=-1)

class WaterFlow(nn.Module):
    """Stacked Affine Coupling Layers"""
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

# --- THERMODYNAMIC LOSS ---

def thermodynamic_loss_shannon(model, x, beta, lambda_reg=1e-6):
    """F = beta * E(x) - lambda * S_reg"""
    
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
            
    # 3. Calculate F (Free Energy)
    F_loss = beta * E_x - lambda_reg * S_reg
    
    return F_loss, E_x.item(), S_reg.item()

# --- ANNEALING SCHEDULE ---
def get_beta(step, total_steps_warmup):
    return min(1.0, step / total_steps_warmup)

# --- TRAINING LOOP ---

if __name__ == '__main__':
    torch.manual_seed(42)
    print(f"Using device: {device}")

    # Load MNIST Data 
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
    NUM_EPOCHS = 10 # Reduced for quick demo
    
    # Thermodynamic Setup
    TOTAL_STEPS = NUM_EPOCHS * len(train_loader)
    WARMUP_STEPS = int(0.5 * TOTAL_STEPS) 
    LAMBDA_REG = 1e-6 

    model = WaterFlow(INPUT_DIM, NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Starting WaterFlow (Thermodynamic Training) ---")
    step_count = 0

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, (data, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"):
            data = data.to(device)
            beta = get_beta(step_count, WARMUP_STEPS)
            
            optimizer.zero_grad()
            F_loss, E_x, S_reg = thermodynamic_loss_shannon(model, data, beta, LAMBDA_REG)
            
            F_loss.backward()
            optimizer.step()
            
            if step_count % 500 == 0:
                BPD = E_x * (np.log2(np.e) / INPUT_DIM) 
                # Print log only every 500 steps for cleaner output
                print(f" | Beta: {beta:.3f} | F Loss: {F_loss.item():.4f} | BPD: {BPD:.4f} | S (Shannon): {S_reg:.2f}")

            step_count += 1
    
    print("\nTraining complete. Model reached thermodynamic equilibrium.")


    # --- VISUALIZATION AND EXPERIMENTS ---
    model.eval()

    # 5.1. GENERATION (Reversibility Check)
    print("\n--- Generating New Samples (Reversibility Proof) ---")
    num_samples = 16
    z_samples = model.prior.sample((num_samples, model.input_dim)).to(device)
    with torch.no_grad():
        x_generated = model.reverse(z_samples)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        img = x_generated[i].cpu().numpy().reshape(28, 28)
        ax.imshow(np.clip(img, 0, 1), cmap='gray')
        ax.axis('off')
    plt.suptitle("Generated Images by WaterFlow (Reversible)")
    plt.show()


    # 5.2. CAPILLARY ACTION ANIMATION (UMAP Flow)
    print("\n--- Starting UMAP Capillary Flow Animation Render ---")

    data_for_flow_vis, _ = next(iter(train_loader))
    data_for_flow_vis = data_for_flow_vis[:5].to(device) 

    flow_trajectory = [] 
    current_state = data_for_flow_vis
    num_samples_to_plot = 5

    # PHASE 1: Forward (X -> Z)
    with torch.no_grad():
        for layer in model.layers:
            current_state, _ = layer(current_state) 
            flow_trajectory.append(current_state.cpu().numpy())

    # PHASE 2: Reverse (Z -> X)
    current_state_reverse = current_state
    with torch.no_grad():
        for layer in reversed(model.layers):
            current_state_reverse = layer.reverse(current_state_reverse)
            flow_trajectory.append(current_state_reverse.cpu().numpy())

    # UMAP Reduction
    all_points_flat = np.concatenate([arr.reshape(-1, model.input_dim) for arr in flow_trajectory], axis=0)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    reduced_trajectory_flat = reducer.fit_transform(all_points_flat)

    num_layers = len(model.layers)
    total_steps = (num_layers * 2) 
    reduced_trajectory = reduced_trajectory_flat.reshape(total_steps, -1, 2)

    # Create Animation Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.get_cmap('hsv', num_samples_to_plot) 
    scatter_plots = []
    for i in range(num_samples_to_plot):
        line, = ax.plot([], [], '-', color=colors(i), alpha=0.6, linewidth=1.5, label=f'Sample {i+1}')
        scatter = ax.scatter([], [], color=colors(i), marker='o', s=100)
        scatter_plots.append((line, scatter))

    # Key points markers
    start_X = reduced_trajectory[0, :, :]
    end_Z_start_R = reduced_trajectory[num_layers - 1, :, :] 
    start_R_end_X = reduced_trajectory[-1, :, :] 

    ax.scatter(start_X[:, 0], start_X[:, 1], marker='s', s=150, color='black', label='Start X (Data Space)')
    ax.scatter(end_Z_start_R[:, 0], end_Z_start_R[:, 1], marker='*', s=200, color='red', label='Z Latent (Reversal Point)')
    ax.scatter(start_R_end_X[:, 0], start_R_end_X[:, 1], marker='D', s=150, color='blue', label='End X (Final State)')
    ax.grid(True)
    ax.legend(loc='upper right')

    # Animation Functions
    def init():
        ax.set_xlim(reduced_trajectory_flat[:, 0].min() - 1, reduced_trajectory_flat[:, 0].max() + 1)
        ax.set_ylim(reduced_trajectory_flat[:, 1].min() - 1, reduced_trajectory_flat[:, 1].max() + 1)
        return [line for line, scatter in scatter_plots] + [scatter for line, scatter in scatter_plots]

    def update(frame):
        for i in range(num_samples_to_plot):
            x_data = reduced_trajectory[:frame+1, i, 0]
            y_data = reduced_trajectory[:frame+1, i, 1]
            scatter_plots[i][0].set_data(x_data, y_data)
            scatter_plots[i][1].set_offsets(reduced_trajectory[frame, i, :])
        
        if frame < num_layers:
            phase = "Forward (X → Z)"
        else:
            phase = "Reverse (Z → X)"
            
        ax.set_title(f"Capillary Flow Animation: {phase} | Layer {frame % num_layers + 1}/{num_layers}")
            
        return [line for line, scatter in scatter_plots] + [scatter for line, scatter in scatter_plots]

    # Render and Display
    anim = animation.FuncAnimation(fig, update, frames=total_steps, init_func=init, interval=150, blit=True)
    HTML(anim.to_jshtml())

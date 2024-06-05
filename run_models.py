import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
import torch.cuda as cuda
import pickle

torch.set_float32_matmul_precision('high')

# Training parameters
steps = 500
interval = 1  # Save the model output every `interval` epochs

# Number of datasets to sample
num_datasets = 10

# Function to generate random dataset
def generate_dataset(seed):
    torch.manual_seed(1212343 + seed)
    x = torch.linspace(-2, 2, 500)
    y0 = torch.special.bessel_j0(20 * x)
    y = y0 + .1 * torch.randn(x.shape)
    x_tensor = x.view(-1, 1).float()
    y_tensor = y.view(-1, 1).float()
    return x_tensor.cuda(), y_tensor.cuda(), y0.cuda()

# List of network architectures
from models import SimpleNet, ExpansionMLP, LearnedActivationMLP, RegluMLP, Kan, RegluExpandMLP, Mix2MLP

# All architectures are 2-hidden layer MLPs with (1, 100, 100, 1) units
architectures = {
    # Params: 2d^2 + O(d)
    "Simple": SimpleNet(d=100),
    # Params: 4kd^2 + O(d)
    "Expanding": ExpansionMLP(d=100, k=3, scale=1/8),
    # Params: 2d^2 + 6dk + O(d)
    "Learned Act": LearnedActivationMLP(d=100, k=3),
    # Params: 4d^2 + O(d)
    "Gated Sine": RegluMLP(d=100, func=torch.sin),
    "Reglu": RegluMLP(d=100, func=torch.relu),
    # Params: 6kd^2 + O(dk)
    "KAN": Kan(d=100, k=3, scale=.5, func=torch.relu),
    # Params: 4kd^2 + O(dk)
    "MoE": Mix2MLP(d=100, k=3, func=torch.relu),
}
architectures = {
    name: net.cuda()
    for name, net in architectures.items()
} | {
    name + "_compiled": torch.compile(net.cuda())
    for name, net in architectures.items()
}

# Dictionary to store loss history, step times, and memory usage
loss_history = {name: [] for name in architectures.keys()}
step_times = {name: [] for name in architectures.keys()}
memory_usage = {name: [] for name in architectures.keys()}

# Training loop for each architecture on each dataset
for dataset_id in range(num_datasets):
    print(f"Training on Dataset {dataset_id+1}")
    x_tensor, y_tensor, _ = generate_dataset(seed=dataset_id)
    for name, net in architectures.items():
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.01)

        losses = []
        # Measure step times and memory usage
        start_time = cuda.Event(enable_timing=True)
        end_time = cuda.Event(enable_timing=True)
        start_time.record()
        with tqdm.tqdm(range(steps), desc=f"Training {name} on Dataset {dataset_id+1}") as pbar:
            for step in pbar:
                net.train()
                optimizer.zero_grad()
                outputs = net(x_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()

                if step % interval == 0:
                    losses.append(loss.item())
                pbar.set_postfix({"loss": loss.item()})
        end_time.record()
        cuda.synchronize()
        step_times[name].append(start_time.elapsed_time(end_time) / steps)
        memory_usage[name].append(torch.cuda.max_memory_allocated() / 1024 / 1024)  # Convert to MB
        loss_history[name].append(losses)

# Save the data to a file
data = {
    "loss_history": loss_history,
    "step_times": step_times,
    "memory_usage": memory_usage
}
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)
print("Data saved to data.pkl")

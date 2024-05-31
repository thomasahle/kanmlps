import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tqdm

# Generate random dataset
np.random.seed(0)
x = torch.linspace(-2, 2, 500)
# y = np.sin(20 * x) + 2 * (x + x**2)
y0 = torch.special.bessel_j0(20 * x)
y = y0 + np.random.normal(0, 0.1, x.shape)
x_tensor = x.view(-1, 1).float()
y_tensor = y.view(-1, 1).float()


# Instantiate the network
from models import SimpleNet, ExpansionMLP, LearnedActivationMLP, RegluMLP, Kan, RegluExpandMLP, Mix2MLP
#net = SimpleNet(d=100)
#net = RegluMLP(d=100, func=torch.sin)
#net = Kan(d=33, k=3)
#net = Kan(d=100, k=3)
net = Mix2MLP(d=100, k=3)
#net = RegluMLP(d=100)
#net = RegluExpandMLP(d=10)
#net = LearnedActivationMLP(d=100, k=10)

# define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

loss = criterion(y0.view(-1, 1).float(), y_tensor)
print(f"Epistemic loss: {loss.item():.4}")

# Training parameters
steps = 1000
interval = 10  # Save the model output every `interval` epochs

# Prepare the figure for plotting
fig, ax = plt.subplots()
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.spines["bottom"].set_color("white")
ax.spines["top"].set_color("white")
ax.spines["left"].set_color("white")
ax.spines["right"].set_color("white")
ax.tick_params(axis="x", colors="white")
ax.tick_params(axis="y", colors="white")
ax.yaxis.label.set_color("white")
ax.xaxis.label.set_color("white")
ax.title.set_color("white")

# Prepare the figure for plotting
ax.plot(x, y, "r.")
ax.set_ylim(-2, 3)

# Training loop
frames = []

with tqdm.tqdm(range(steps)) as pbar:
    for step in pbar:
        net.train()
        optimizer.zero_grad()
        outputs = net(x_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if step % interval == 0:
            net.eval()
            with torch.no_grad():
                pred_y = net(x_tensor).numpy()
                frame = ax.plot(x, pred_y, "y-")
                title = ax.text(
                    0.5,
                    1.05,
                    f"Step {step}, Loss: {loss.item():.4f}",
                    size=plt.rcParams["axes.titlesize"],
                    ha="center",
                    transform=ax.transAxes,
                    color="white",
                )
                frames.append(frame + [title])

        pbar.set_postfix({"loss": loss.item()})

ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True)
ani.save("fit_animation.mp4", writer="ffmpeg")

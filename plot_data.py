import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.patches import Patch

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--plot-type", type=str, default="loss", help="Type of plot: 'loss' or 'error_bars'"
)
parser.add_argument(
    "--background-color",
    type=str,
    default="white",
    help="Background color: 'white' or 'black'",
)
parser.add_argument(
    "--skip-items", type=int, default=0, help="Number of items to skip from the data"
)
args = parser.parse_args()

architectures_list = ["Expanding", "Learned Act", "Reglu", "KAN", "MoE"]


# Load the data from the file
with open("data.pkl", "rb") as f:
    data = pickle.load(f)

loss_history = data["loss_history"]
step_times = data["step_times"]
memory_usage = data["memory_usage"]

# Convert from ms/step to steps/s
step_times = {name: 1000 / np.array(times) for name, times in step_times.items()}

# Skip the specified number of items from the data
loss_history = {
    name: losses[args.skip_items :] for name, losses in loss_history.items()
}
step_times = {name: times[args.skip_items :] for name, times in step_times.items()}
memory_usage = {
    name: memory[args.skip_items :] for name, memory in memory_usage.items()
}

# Compute mean and standard deviation of losses, step times, and memory usage
mean_losses = {name: np.mean(losses, axis=0) for name, losses in loss_history.items()}
std_losses = {name: np.std(losses, axis=0) for name, losses in loss_history.items()}
mean_step_times = {name: np.mean(times) for name, times in step_times.items()}
std_step_times = {name: np.std(times) for name, times in step_times.items()}
mean_memory_usage = {name: np.mean(memory) for name, memory in memory_usage.items()}
std_memory_usage = {name: np.std(memory) for name, memory in memory_usage.items()}

# Compute median and 25th percentile of losses
median_losses = {name: np.median(losses, axis=0) for name, losses in loss_history.items()}
q25_losses = {name: np.percentile(losses, 25, axis=0) for name, losses in loss_history.items()}
q75_losses = {name: np.percentile(losses, 75, axis=0) for name, losses in loss_history.items()}

#colors = plt.cm.tab10(np.linspace(0, 1, len(architectures_list)))
colors = plt.cm.tab10(range(len(architectures_list)))

if args.plot_type == "loss":
    # Plotting the loss history with fill_between
    fig, ax = plt.subplots()
    x_values = range(0, 500, 1)
    for i, name in enumerate(architectures_list):
        median = median_losses[name]
        q25 = q25_losses[name]
        q75 = q75_losses[name]  # Calculate the 75th percentile
        ax.plot(x_values, median, label=name, color=colors[i])
        ax.fill_between(x_values, q25, q75, alpha=0.2, color=colors[i], edgecolor=None)

    # Determine y-axis limits by ignoring outliers
    all_losses = np.concatenate(
        [np.concatenate(losses) for losses in loss_history.values()]
    )
    # q1, q3 = np.percentile(all_losses, [25, 75])
    # iqr = q3 - q1
    # lower_bound = q1 - 1 * iqr
    # upper_bound = q3 + 2.5 * iqr
    ax.set_ylim(0.005, 0.06)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("Loss of MLP types on Bessel0 Function")

    if args.background_color == "black":
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        ax.spines["bottom"].set_color("white")
        ax.spines["top"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["right"].set_color("white")
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")

    # Save the plot to a file
    plt.box(on=None)
    plt.savefig(
        "loss_comparison_with_variance.png", facecolor=fig.get_facecolor(), dpi=300
    )
    print("Plot saved to loss_comparison_with_variance.png")


elif args.plot_type == "error_bars":
    x = np.arange(len(architectures_list))
    width = 0.35

    for file_name, data_dict, ylabel, title in zip(
        ["step_times.png", "memory_usage.png"],
        #[step_times, memory_usage],
        [mean_step_times, mean_memory_usage],
        ["Steps / Second", "Memory Usage (MB)"],
        [
            "MLP Speed After torch.compile",
            "Memory Usage for Different Architectures",
        ],
    ):
        fig, ax = plt.subplots()

        for i, arch in enumerate(architectures_list):
            non_compiled_name = arch
            compiled_name = arch + "_compiled"

            non_compiled_color = (*colors[i][:3], 0.7)
            compiled_color = (*colors[i][:3], 0.4)
            ax.bar(
                x[i] - width / 2,
                data_dict[non_compiled_name],
                width,
                alpha=0.7,
                yerr=std_step_times[non_compiled_name],
                color=colors[i],
                #ecolor="white" if args.background_color == "black" else "black",
            )
            ax.bar(
                x[i] + width / 2,
                data_dict[compiled_name],
                width,
                alpha=0.4,
                yerr=std_step_times[compiled_name],
                color=colors[i],
                hatch="/",
                #ecolor="white" if args.background_color == "black" else "black",
            )

        # ax.set_xlabel("Architecture")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(architectures_list, rotation=45, ha="right")

        # Create custom legend handles and labels
        ax.legend(
            handles=[
                Patch(facecolor=colors[0], alpha=0.7, label="Normal"),
                Patch(facecolor=colors[0], alpha=0.4, hatch="//", label="Compiled"),
            ]
        )

        if args.background_color == "black":
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
            fig.patch.set_facecolor("black")

        plt.box(on=None)
        plt.tight_layout()
        plt.savefig(file_name, facecolor=fig.get_facecolor(), dpi=300)
        print(f"Plot saved to {file_name}")
        plt.close()

else:
    print("Invalid plot type. Please choose 'loss' or 'error_bars'.")

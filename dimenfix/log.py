import matplotlib.pyplot as plt
import time
import os


def print_layout(y, label, title=None):
    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=label,
                cmap='tab10', edgecolors='face', linewidths=0.5, s=12)
    plt.grid(linestyle='dotted')

    if title is not None:
        plt.title(title)

    os.makedirs("./log", exist_ok=True)

    timestamp = int(time.time() * 1000.0)
    filename = "./log/log_print_layout_" + str(timestamp) + ".png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

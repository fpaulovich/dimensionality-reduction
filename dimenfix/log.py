# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

import matplotlib.pyplot as plt
import time
import os


def clean(active=True):
    if active is True:
        os.makedirs("./log", exist_ok=True)

        for filename in os.listdir("./log"):
            if filename.endswith('.png'):
                os.remove(os.path.join("./log", filename))


def print_layout(y, label, title=None, active=True):
    if active is True:
        plt.figure()
        plt.scatter(y[:, 0], y[:, 1], c=label,
                    cmap='tab10', edgecolors='face', linewidths=0.5, s=12)
        plt.grid(linestyle='dotted')
        plt.colorbar()

        if title is not None:
            plt.title(title)

        os.makedirs("./log", exist_ok=True)

        timestamp = int(time.time() * 1000.0)
        filename = "./log/log_print_layout_" + str(timestamp) + ".png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

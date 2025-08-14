import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def plot(save_fp: str, x_vals: list, y_vals: list, title: str, x_label: str, y_label: str) -> None:
    ax = plt.figure().gca()
    # make sure the x-axis uses integers rather than floats
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    title = title.replace("_", " ")
    plt.title(title)
    plt.plot(x_vals, y_vals)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(save_fp)
    plt.clf()
    plt.close()


def plot_histogram(vals_: list, title: str, y_label: str, x_label: str, save_fp: str,
                   n_bins: int = 10) -> None:
    vals = np.array(vals_, dtype=float)
    plt.hist(vals, density=False, bins=n_bins)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(save_fp)
    plt.clf()
    plt.close()

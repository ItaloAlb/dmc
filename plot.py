# plot.py
import sys
import numpy as np
import matplotlib.pyplot as plt

def plot_energies(filename):
    data = np.loadtxt(filename)
    steps = data[:, 0]
    block_energy = data[:, 1]
    ref_energy   = data[:, 2]
    mean_energy  = data[:, 3]

    plt.figure(figsize=(8,5))
    plt.plot(steps, ref_energy,   label="Reference Energy", linestyle="-", alpha=0.7)
    plt.plot(steps, block_energy, label="Block Energy", linestyle="-", alpha=0.7)
    plt.plot(steps, mean_energy,  label="Mean Energy", linestyle="-", alpha=0.7)

    plt.xlabel("Step")
    plt.ylabel("Energy (a.u.)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_population(filename):
    data = np.loadtxt(filename)
    steps = data[:, 0]
    population = data[:, 4]

    plt.figure(figsize=(8,5))
    plt.plot(steps, population, color="tab:red", label="Population")

    plt.xlabel("Step")
    plt.ylabel("Number of Walkers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    filename = "dmc.dat"
    if len(sys.argv) == 1:
        plot_energies(filename)
        plot_population(filename)

    elif len(sys.argv) == 3:
        func = sys.argv[1]
        filename = sys.argv[2]
        
        if func == "energy":
            plot_energies(filename)
        elif func == "population":
            plot_population(filename)
        else:
            print(f"Unkown function: {func}")
    else:
        print("Usage:")
        print("  python plot.py                -> energy and population plot (default file dmc.dat)")
        print("  python plot.py energy file  -> energy plot")
        print("  python plot.py population file -> population plot")

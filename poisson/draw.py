import matplotlib.pyplot as plt
import numpy as np
import sys

def draw(path, model, save=True):
    """
    Draw 2D Poisson problem data.

    Parameters:
    - path: str, directory containing the solution files
    - model: str, base filename for the solution files
    - save: bool, whether to save the plot as a file or display it
    """
    # Read grid size and boundary value
    nx, ny, _ = np.genfromtxt(f"{path}/{model}.info", delimiter=',', missing_values='-')
    nx, ny = int(nx), int(ny)
    u = np.reshape(np.loadtxt(f"{path}/{model}.dat", skiprows=2), (ny, nx))

    # Generate grid for plotting
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)

    # Plot the solution
    fig, ax = plt.subplots()
    pc = ax.pcolormesh(x, y, u, shading='auto', cmap='viridis')
    fig.colorbar(pc, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Solution of Poisson Equation (in 2D)")

    # Show or save the plot
    if save:
        plt.savefig(f"{path}/{model}.png")
    else:
        plt.show()


def draw_3d(path, model, save=True):
    """
    Draw 3D Poisson problem data.

    Parameters:
    - path: str, directory containing the solution files
    - model: str, base filename for the solution files
    - save: bool, whether to save the plot as a file or display it
    """
    # Read grid size and boundary value
    nx, ny, _ = np.genfromtxt(f"{path}/{model}.info", delimiter=',', missing_values='-')
    nx, ny = int(nx), int(ny)
    u = np.reshape(np.loadtxt(f"{path}/{model}.dat", skiprows=2), (ny, nx))

    # Generate grid for 3D plotting
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    # Plot the solution in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, u, cmap='viridis', edgecolor='k')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.set_title("Solution of Poisson Equation (in 3D)")

    # Show or save the plot
    if save:
        plt.savefig(f"{path}/{model}_3d.png")
    else:
        plt.show()


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python draw.py  <FILENAME with out .dat .info>")
        sys.exit(1)

    filename = sys.argv[1]

    draw("./", filename)
    draw_3d("./", filename)

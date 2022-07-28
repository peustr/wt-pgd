import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa


def create_figure(x, y, z, z_limits=None, savefig_path=None):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(20, 160)

    ax.set_xlabel('Grad.', labelpad=15, fontsize=20)
    ax.set_xlim(-0.031, 0.031)
    ax.set_xticks([-0.031, -0.0155, 0, 0.0155, 0.031])
    ax.set_xticklabels(["-ε", "-ε/2", 0, "ε/2", "ε"])
    ax.tick_params(axis='x', labelsize=18)

    ax.set_ylabel('Grad. orth.', labelpad=15, fontsize=20)
    ax.set_ylim(-0.031, 0.031)
    ax.set_yticks([-0.031, -0.0155, 0, 0.0155, 0.031])
    ax.set_yticklabels(["-ε", "-ε/2", 0, "ε/2", "ε"])
    ax.tick_params(axis='y', labelsize=18)

    if z_limits is None:
        z_min, z_max = 0, max(z)
    else:
        z_min, z_max = z_limits

    ax.set_zlabel('Loss', labelpad=0, rotation="vertical", fontsize=20)
    ax.set_zlim(z_min, z_max)
    ax.set_zticks([z_min, z_max])
    ax.tick_params(axis='z', labelsize=18, pad=10)

    ax.plot_trisurf(x, y, z, cmap=cm.Purples, linewidth=0, antialiased=False, edgecolors='black')
    ax.xaxis.pane.set_facecolor('lavender')
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_facecolor('lavender')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_facecolor('lavender')
    ax.zaxis.pane.set_edgecolor('black')
    ax.grid(False)

    if savefig_path is not None:
        plt.tight_layout()
        plt.savefig(savefig_path)
    return fig, ax

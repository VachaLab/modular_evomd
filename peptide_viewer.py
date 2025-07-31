# === peptide_viewer.py ===
from sequence import Sequence
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np
import argparse


def get_arguments() -> argparse.Namespace:
    """
    Peptide viewer arguments from command-line
    """
    parser = argparse.ArgumentParser(description='Peptide 3D-viewer :)')

    # Input file with sequences and additional options
    parser.add_argument(
        '-s', '--sequence',
        help='Sequence to plot',
        required=True
    )
    parser.add_argument(
        '-p', '--parameter',
        help='Parameter to plot (hydrophobicity, charge, faces)',
        default='hydrophobicity'
    )
    parser.add_argument(
        '-l', '--letters',
        help='Include residue letters in the plot',
        action='store_true',
    )
    parser.add_argument(
        '-d', '--two-d',
        help='2D plot',
        action='store_true',
    )
    parser.add_argument(
        '-f', '--show-slices',
        help='Show slices?',
        action='store_true'
    )
    parser.add_argument(
        '-a', '--slice-angle',
        help='Slice angle to split faces. Default=180°',
        default=180, type=float
    )
    parser.add_argument(
        '-m', '--cmap',
        help='Name of the matplotlib colormap. Default=viridis',
        default='viridis', type=str
    )

    args = parser.parse_args()

    return args

def set_colors(args: argparse, seq: Sequence):
    cmap = cm.get_cmap(args.cmap)
    if args.parameter == 'hydrophobicity':
        res_hp = np.array([k.hydrophobicity for k in seq.residues])  # hydrophobicity
    elif args.parameter == 'charge':
        res_hp = np.array([k.charge for k in seq.residues]+[1, -1])  # charges: always include 1 and -1
    elif args.parameter == 'faces':
        p_face, _ = seq.get_faces(phi=args.slice_angle)
        res_hp = np.array([1 if k.index in p_face else -1 for k in seq.residues])  # faces
    else:
        print(f'Parameter {args.parameter} unrecognized')
        exit(1)
    norm = (res_hp - np.min(res_hp)) / (np.max(res_hp) - np.min(res_hp))
    colors = [cmap(h) for h in norm]
    for res in seq.residues:
        res.color = colors[res.index]
    seq.cmap = cmap
    seq.norm_colors = norm
    seq.colors = colors
    return seq

def plot_peptide(args: argparse, seq: Sequence, fig, ax):
    # plot hydrophobic vector
    if args.two_d:
        ax.plot([0,2], [0,0], lw=2, c='deepskyblue', zorder=1)
    else:
        ax.plot([0,2], [0,0], [0,0], lw=2, c='deepskyblue')

    # plot peptide
    i = 0
    visited = []
    for res in seq.residues:
        df = 1
        if args.two_d:
            dis_mat = [False if np.linalg.norm(np.array([res.x, res.y])-k)>0.05 else True for k in visited]
            if len(visited) > 0 and any(dis_mat):
                df = 1.2
            ax.scatter(res.x*df, res.y*df, c=[res.color], s=80, alpha=1, zorder=3)
            visited.append(np.array([res.x, res.y]))
        else:
            ax.scatter(res.x, res.y, res.z, c=[res.color], s=80, alpha=0.8)
        if args.letters:
            scale = 1.4
            fontsize = 9
            letter_pos = [res.x*scale*df, res.y*scale*df, res.z]
            if args.two_d:
                ax.text(letter_pos[0], letter_pos[1], res.letter + f'{res.index + 1}', size=fontsize, ha='center', va='center')
            else:
                ax.text(letter_pos[0], letter_pos[1], letter_pos[2], res.letter + f'{res.index + 1}', size=fontsize, ha='center', va='center')
        if i > 0:
            if args.two_d:
                ax.plot([seq.residues[i-1].x, res.x], [seq.residues[i-1].y, res.y], c='gray', lw=1, zorder=1)
            else:
                ax.plot([seq.residues[i-1].x, res.x], [seq.residues[i-1].y, res.y], [seq.residues[i-1].z, res.z], c='gray', lw=1)
        i += 1
    
    # plot slices
    if args.show_slices:
        phi = args.slice_angle * np.pi / 180
        extension = 1.5
        border = np.array([np.cos(phi/2)*extension, np.sin(phi/2)*extension, 0.])
        if args.two_d:
            ax.plot([0, border[0]], [0,  border[1]], lw=2, c='lightgreen', zorder=1)
            ax.plot([0, border[0]], [0, -border[1]], lw=2, c='lightgreen', zorder=1)
        else:
            # positions
            points = seq.get_positions()
            # get maximum Z value
            max_z = np.max(points[:,2])*1.2
            # create meshgrid
            s = np.linspace(0, np.linalg.norm(border), 20)
            t = np.linspace(-max_z, max_z, 20)
            S, T = np.meshgrid(s, t)
            # get points in plane
            X_plane   =  border[0] * S
            Y_plane_1 =  border[1] * S
            Y_plane_2 = -border[1] * S
            Z_plane   =  T
            ax.plot_surface(X_plane, Y_plane_1, Z_plane, alpha=0.2, color='lightgreen', edgecolor='none')
            ax.plot_surface(X_plane, Y_plane_2, Z_plane, alpha=0.2, color='lightgreen', edgecolor='none')
    return fig, ax

def create_figure(args, seq):
    if args.two_d:
        # 2D figure
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
    else:
        # 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        points = seq.get_positions()
        # get maximum Z value
        max_z = np.max(points[:,2])
        ax.set_zlim([-max_z, max_z])  # Z range
        ax.grid(False)
    ax.set_axis_off()
    ax.set_xlim([-2.5, 2.5])  # X range
    ax.set_ylim([-2.5, 2.5])  # Y range
    ax.set_title(r'$\alpha$-Helix')
    return fig, ax

def set_legends(args: argparse, seq: Sequence, fig, ax):
    if args.parameter == 'hydrophobicity':
        # Create a colorbar for hydrophobicity
        res_hp = np.array([k.hydrophobicity for k in seq.residues])
        sm = plt.cm.ScalarMappable(cmap=seq.cmap, norm=plt.Normalize(vmin=np.min(res_hp), vmax=np.max(res_hp)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Hydrophobicity')
    elif args.parameter == 'charge':
        # Create a legend
        charge_colors = {'Positive': seq.cmap(1.0), 'Neutrum': seq.cmap(0.5), 'Negative': seq.cmap(0)}
        # Create Line2D objects for the legend
        charge_lines = [Line2D([0], [0], marker='s', color=color, markerfacecolor=color, markersize=5, linewidth=0, label=f'{charge_type}') 
                        for charge_type, color in charge_colors.items()]
        # Add legend to the plot
        ax.legend(handles=charge_lines, loc='upper center', title="Charge", frameon=False, facecolor='none', ncol=3)
    elif args.parameter == 'faces':
        # Create a legend
        charge_colors = {'Positive': seq.cmap(1.0), 'Negative': seq.cmap(0)}
        # Create Line2D objects for the legend
        charge_lines = [Line2D([0], [0], marker='s', color=color, markerfacecolor=color, markersize=5, linewidth=0, label=f'{charge_type}') 
                        for charge_type, color in charge_colors.items()]
        # Add legend to the plot
        ax.legend(handles=charge_lines, loc='upper center', title="Faces", frameon=False, facecolor='none', ncol=2)
    else:
        print(f'Parameter {args.parameter} unrecognized')
        exit(1)
    return fig, ax

def main():
    args = get_arguments()

    seq = Sequence(args.sequence)

    # create adequate figure
    fig, ax = create_figure(args, seq)

    # set colors
    seq = set_colors(args, seq)

    # plot peptide
    fig, ax = plot_peptide(args, seq, fig, ax)

    # set proper legends
    fig, ax = set_legends(args, seq, fig, ax)

    plt.show()


if __name__ == '__main__':
    main()


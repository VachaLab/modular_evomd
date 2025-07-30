from sequence import Sequence
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import argparse


def get_arguments() -> argparse.Namespace:
    """
    Parse and return command-line arguments for the Evo-MD-FE-DB agent.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
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
        help='Parameter to plot (residue hydrophobicity or residue charge)',
        default='hydrophobicity'
    )
    parser.add_argument(
        '-l', '--letters',
        help='Include residue letters in the plot',
        action='store_true',
    )

    args = parser.parse_args()
    return args


def get_colors(sequence, method='hydrophobicity', cmap='bwr'):
    cmap = cm.get_cmap(cmap)
    if method == 'hydrophobicity':
        res_hp = np.array([k.hydrophobicity for k in sequence.residues])  # hydrophobicity
        # normalize [0,1]
        norm = (res_hp - np.min(res_hp)) / (np.max(res_hp) - np.min(res_hp))
        # 2 * (res_hp - x_min) / (x_max - x_min) - 1
        colors = [cmap(h) for h in norm]
    else:
        res_hp = np.array([k.charge for k in sequence.residues])  # charges
        norm = (res_hp - np.min(res_hp)) / (np.max(res_hp) - np.min(res_hp))
        colors = [cmap(h) for h in norm]
    return colors


def main():
    args = get_arguments()

    seq = Sequence(args.sequence)

    # positions
    points = seq.get_positions()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # translate to 0,0
    z_center = np.mean( points[:,2] )
    center = np.array([0, 0, z_center])
    points -= center

    # get maximum Z value
    max_z = np.max(points[:,2])

    # plot hydrophobic vector
    ax.plot([0,2], [0,0], [0,0], lw=2)

    # generate color list
    colors = get_colors(seq, method=args.parameter)

    # plot
    i = 0
    for point, letter in zip(points, seq):
        ax.scatter(point[0], point[1], point[2], c=[colors[i]], s=80, alpha=0.8)
        if args.letters:
            scale = 1.5
            letter_pos = [point[0]*scale, point[1]*scale, point[2]]
            ax.text(letter_pos[0], letter_pos[1], letter_pos[2], letter)
        if i > 0:
            ax.plot([points[i-1][0], point[0]], [points[i-1][1], point[1]], [points[i-1][2], point[2]], c='gray', lw=1)
        i += 1

    ax.set_xlim([-3, 3])  # Establecer el rango para el eje X
    ax.set_ylim([-3, 3])  # Establecer el rango para el eje Y
    ax.set_zlim([-max_z, max_z])  # Establecer el rango para el eje Z

    ax.set_title(r'$\alpha$-Helix')
    plt.show()


if __name__ == '__main__':
    main()


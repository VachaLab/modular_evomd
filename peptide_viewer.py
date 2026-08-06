# === peptide_viewer.py ===
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from sequence import Sequence
from utils import int_to_roman
from sequence_geometry import *


# --- Argument validation ---------------------------------------------------

def _validate_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser = None,
) -> None:
    """
    Validates conditional argument dependencies.
    Uses parser.error() in CLI context, raises ValueError in import/notebook context.
    """
    def _error(msg: str) -> None:
        if parser:
            parser.error(msg)
        else:
            raise ValueError(msg)

    if args.slice_angle != 180.0 and not args.show_slices and args.parameter != 'faces':
        _error('slice_angle requires show_slices=True or parameter="faces".')
    if args.sections != 18 and not args.show_sections:
        _error('sections requires show_sections=True.')
    if args.show_slices and args.show_sections:
        _error('show_slices and show_sections are mutually exclusive.')


# --- Color setup -----------------------------------------------------------

def set_colors(
    args: argparse.Namespace,
    seq: Sequence,
    positions: np.ndarray,
    cmap,
) -> list:
    """
    Computes per-residue colors based on the selected parameter.
    For 'charge', +1 and -1 are added as anchors to fix colormap endpoints.
    Returns a list of RGBA colors indexed by residue index.
    """
    if args.parameter == 'hydrophobicity':
        anchor = np.array([r.hydrophobicity for r in seq.residues])
        residue_values = anchor
    elif args.parameter == 'charge':
        # Anchor values ensure the colormap always spans the full charge range
        anchor = np.array([r.charge for r in seq.residues] + [1.0, -1.0])
        residue_values = np.array([r.charge for r in seq.residues])
    elif args.parameter == 'faces':
        p_face, _ = get_faces(positions, seq, args.slice_angle, ref_angle=np.pi * 3/2)
        anchor = np.array([1 if r.index in p_face else -1 for r in seq.residues])
        residue_values = anchor
    else:
        raise ValueError(f"Unrecognized parameter '{args.parameter}'.")

    v_min, v_max = np.min(anchor), np.max(anchor)
    span = v_max - v_min if v_max != v_min else 1.0
    norm = (residue_values - v_min) / span
    return [cmap(float(n)) for n in norm]


# --- Figure creation -------------------------------------------------------

def create_figure(args: argparse.Namespace, positions: np.ndarray) -> tuple:
    """Creates and configures the matplotlib figure and axes."""
    if args.two_d:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        max_z = np.max(np.abs(positions[:, 2])) * 1.2
        ax.set_zlim([-max_z, max_z])
        if args.remove_background:
            ax.grid(False)

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    if args.remove_background:
        ax.set_axis_off()
    return fig, ax


# --- Plotting functions ----------------------------------------------------

def plot_peptide(
    args: argparse.Namespace,
    seq: Sequence,
    positions: np.ndarray,
    colors: list,
    fig,
    ax,
) -> tuple:
    """
    Draws residues as scatter points connected by lines.
    Applies a slight radial offset in 2D when residues overlap.
    """
    # Hydrophobic moment vector arrow pointing toward -Y
    if args.two_d:
        # Hydrophobicity vector:
        # Change from simple line
        # ax.plot([0, 0], [0, -1.5], lw=2, c='deepskyblue', zorder=1)
        # to array.
        ax.arrow(0, 0, 0, -1.5, color='deepskyblue', zorder=1, shape='full', lw=2, head_width=0.08, head_length=0.15)
    else:
        ax.plot([0, 0], [0, -1.5], [0, 0], lw=2, c='deepskyblue')

    visited_xy = []
    for i, (res, pos) in enumerate(zip(seq.residues, positions)):
        x, y, z = pos
        df = 1.

        if args.two_d:
            overlap = [np.linalg.norm(np.array([x, y]) - v) < 0.05 for v in visited_xy]
            if visited_xy and any(overlap):
                df = 1.34
            ax.scatter(x * df, y * df, c=[colors[res.index]], s=500, alpha=1, zorder=3)
            visited_xy.append(np.array([x, y]))
        else:
            ax.scatter(x, y, z, c=[colors[res.index]], s=100, alpha=0.8)

        if not args.noletters:
            # Brightness of background color
            # Simple brightness calculation
            brightness = 0.299 * colors[res.index][0] + 0.587 * colors[res.index][1] + 0.114 * colors[res.index][2]
            text_color = 'white' if brightness < 0.5 else 'black'
            # Place letters        
            scale = 1.4
            lx, ly = x * df, y * df
            label = res.letter + str(res.index + 1)
            if args.two_d:
                ax.text(lx, ly, label, size=9, ha='center', va='center', c=text_color)
            else:
                ax.text(lx * scale, ly * scale, z, label, size=9, ha='center', va='center')

        if i > 0:
            prev = positions[i - 1]
            if args.two_d:
                ax.plot([prev[0], x], [prev[1], y], c='gray', lw=1, zorder=1)
            else:
                ax.plot([prev[0], x], [prev[1], y], [prev[2], z], c='gray', lw=1)

    return fig, ax


def plot_slices(
    args: argparse.Namespace,
    positions: np.ndarray,
    fig,
    ax,
) -> tuple:
    """
    Draws slice lines (2D) or planes (3D) at +-phi/2 from the -Y axis.
    """
    phi = np.deg2rad(args.slice_angle)
    extension = 1.5
    base_angle = -np.pi / 2
    angle_pos = base_angle + phi / 2
    angle_neg = base_angle - phi / 2
    border_pos = np.array([np.cos(angle_pos), np.sin(angle_pos)])
    border_neg = np.array([np.cos(angle_neg), np.sin(angle_neg)])

    if args.two_d:
        ax.plot([0, border_pos[0] * extension], [0, border_pos[1] * extension],
                lw=2, c='lightgreen', zorder=1)
        ax.plot([0, border_neg[0] * extension], [0, border_neg[1] * extension],
                lw=2, c='lightgreen', zorder=1)
    else:
        max_z = np.max(np.abs(positions[:, 2])) * 1.2
        s = np.linspace(0, extension, 20)
        t = np.linspace(-max_z, max_z, 20)
        S, T = np.meshgrid(s, t)
        for border in [border_pos, border_neg]:
            X_plane = border[0] * S
            Y_plane = border[1] * S
            ax.plot_surface(X_plane, Y_plane, T, alpha=0.2,
                            color='lightgreen', edgecolor='none')

    return fig, ax


def plot_sections(
    args: argparse.Namespace,
    positions: np.ndarray,
    fig,
    ax,
) -> tuple:
    """
    Draws n equal angular sections as dashed boundary lines (2D) or planes (3D).
    Section 0 bisector points toward -Y. Labels use Roman numerals (2D only).
    """
    n = args.sections
    segment_angle = 2 * np.pi / n
    bisector_0 = -np.pi / 2
    extension = 1.5
    label_radius = 0.5

    for i in range(n):
        boundary_angle = bisector_0 - segment_angle / 2 + i * segment_angle
        bx = np.cos(boundary_angle) * extension
        by = np.sin(boundary_angle) * extension

        if args.two_d:
            ax.plot([0, bx], [0, by], 'k--', linewidth=0.8, alpha=0.5, zorder=1)
        else:
            max_z = np.max(np.abs(positions[:, 2])) * 1.2
            s = np.linspace(0, extension, 20)
            t = np.linspace(-max_z, max_z, 20)
            S, T = np.meshgrid(s, t)
            X_plane = (bx / extension) * S
            Y_plane = (by / extension) * S
            ax.plot_surface(X_plane, Y_plane, T, alpha=0.08,
                            color='gray', edgecolor='none')

        # Section label at bisector center (2D only)
        if args.two_d:
            bisector_angle = bisector_0 + i * segment_angle
            lx = np.cos(bisector_angle) * label_radius
            ly = np.sin(bisector_angle) * label_radius
            ax.text(lx, ly, int_to_roman(i + 1),
                    ha='center', va='center', fontsize=7, alpha=0.7)

    return fig, ax


def set_title(
    args: argparse.Namespace,
    seq: Sequence,
    hm_scalar: float,
    hi_value: float,
    ax,
) -> None:
    """Builds and sets the plot title based on active display options."""
    title = r'$\alpha$-Helix'
    extras = []
    if args.print_hm:
        if args.av_hm:
            pretit = r'<$\mu$H>'
        else:
            pretit = r'$\mu$H'
        extras.append(f'{pretit}: {round(hm_scalar, 3)}')
    if args.print_hi:
        if args.av_hm:
            pretit = '<Hi>'
        else:
            pretit = 'Hi'
        extras.append(f'{pretit}: {hi_value}')
    if args.print_charge:
        extras.append(f'Ch: {seq.charge}')
    if extras:
        title += '\n'
        title += '  ' + '  '.join(extras)
    ax.set_title(title)


def set_legends(
    args: argparse.Namespace,
    seq: Sequence,
    positions: np.ndarray,
    cmap,
    fig,
    ax,
) -> tuple:
    """Adds colorbar or discrete legend based on the active parameter."""
    if args.parameter == 'hydrophobicity':
        res_hp = np.array([r.hydrophobicity for r in seq.residues])
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=np.min(res_hp), vmax=np.max(res_hp))
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Hydrophobicity')

    elif args.parameter == 'charge':
        charge_colors = {
            'Positive': cmap(1.0),
            'Neutral':  cmap(0.5),
            'Negative': cmap(0.0),
        }
        handles = [
            Line2D([0], [0], marker='s', color=c, markerfacecolor=c,
                   markersize=5, linewidth=0, label=label)
            for label, c in charge_colors.items()
        ]
        ax.legend(handles=handles, loc='upper center', title='Charge',
                  frameon=False, ncol=3)

    elif args.parameter == 'faces':
        face_colors = {
            'Hydrophobic (+)': cmap(1.0),
            'Hydrophilic (-)': cmap(0.0),
        }
        handles = [
            Line2D([0], [0], marker='s', color=c, markerfacecolor=c,
                   markersize=5, linewidth=0, label=label)
            for label, c in face_colors.items()
        ]
        ax.legend(handles=handles, loc='upper center', title='Faces',
                  frameon=False, ncol=2)

    return fig, ax


# --- Core pipeline ---------------------------------------------------------

def _run_pipeline(args: argparse.Namespace) -> dict:
    """
    Executes the full visualization pipeline given a populated Namespace.
    Shared by both main() and plot_sequence().
    
    Returns a dictionary with sequence properties (hm, hi, charge)
    """
    seq = Sequence(args.sequence, h_scale=args.h_scale) if isinstance(args.sequence, str) else args.sequence
    cmap = plt.get_cmap(args.cmap)

    # Compute geometry before alignment to preserve the HM scalar magnitude
    positions = compute_helix_positions(seq)
    hm_vector = compute_hm_vector(seq, positions)
    hm_scalar = compute_hm_scalar(seq, positions, average=args.av_hm)
    hi_value = compute_hi(seq, average=args.av_hm)
    positions = align_to_minus_y(positions, hm_vector)

    colors = set_colors(args, seq, positions, cmap)
    fig, ax = create_figure(args, positions)
    set_title(args, seq, hm_scalar, hi_value, ax)
    fig, ax = plot_peptide(args, seq, positions, colors, fig, ax)

    if args.show_slices:
        fig, ax = plot_slices(args, positions, fig, ax)
    elif args.show_sections:
        fig, ax = plot_sections(args, positions, fig, ax)

    fig, ax = set_legends(args, seq, positions, cmap, fig, ax)

    if args.save:
        plt.savefig(args.out, dpi=300)

    if not args.noshow:
        plt.show()
    
    return {'hm': hm_scalar, 'hi': hi_value, 'charge': seq.charge, 'npos': seq.pos_res, 'nneg': seq.neg_res}


# --- Public API for notebook / import usage --------------------------------

def plot_sequence(
    sequence: "str | Sequence",
    parameter: str = 'hydrophobicity',
    noletters: bool = False,
    h_scale: str = 'eisenberg',
    av_hm: bool = False,
    print_hm: bool = False,
    print_hi: bool = False,
    print_charge: bool = False,
    two_d: bool = True,
    three_d: bool = False,
    show_slices: bool = False,
    slice_angle: float = 180.0,
    show_sections: bool = False,
    sections: int = 18,
    cmap: str = 'viridis',
    remove_background: bool = False,
    save: bool = False,
    out: str = 'output.png',
    noshow: bool = False,
) -> None:
    """
    Visualizes a peptide as an alpha-helix projection.
    Accepts a sequence string or a Sequence object.
    All parameters mirror the CLI arguments and can be freely set from a notebook.

    Parameters
    ----------
    sequence         : Peptide sequence as string or Sequence object.
    parameter        : Coloring scheme: 'hydrophobicity', 'charge', or 'faces'.
    noletters        : No show residue letters and indices on the plot.
    h_scale          : Define hydrophobicity scale.
    av_hm            : Average htdrophobic moment <mH> as done by HeliQuest(R)
    print_hm         : Show hydrophobic moment value in the title.
    print_hi         : Show hydrophobic index value in the title.
    print_charge     : Show net charge in the title.
    two_d            : 2D helical wheel projection (default).
    three_d          : 3D helix plot. Overrides two_d if True.
    show_slices      : Draw slice lines/planes. Mutually exclusive with show_sections.
    slice_angle      : Slice angle in degrees. Requires show_slices or parameter='faces'.
    show_sections    : Draw equal angular sections. Mutually exclusive with show_slices.
    sections         : Number of angular sections. Requires show_sections.
    cmap             : Matplotlib colormap name.
    remove_background: Remove figure background and axes.
    save             : Save the figure to a file.
    out              : Output filename when save=True.
    """
    args = argparse.Namespace(
        sequence=sequence,
        parameter=parameter,
        noletters=noletters,
        h_scale=h_scale,
        av_hm=av_hm,
        print_hm=print_hm,
        print_hi=print_hi,
        print_charge=print_charge,
        two_d=not three_d,
        three_d=three_d,
        show_slices=show_slices,
        slice_angle=slice_angle,
        show_sections=show_sections,
        sections=sections,
        cmap=cmap,
        remove_background=remove_background,
        save=save,
        out=out,
        noshow=noshow,
    )
    _validate_args(args, parser=None)
    _run_pipeline(args)


# --- Argument parsing ------------------------------------------------------

def get_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the peptide viewer."""
    parser = argparse.ArgumentParser(
        description='Alpha-helix peptide viewer. --sequence is required.'
    )
    parser.add_argument(
        '-s', '--sequence',
        help='Peptide sequence to plot (single-letter amino acid codes).',
        required=True,
    )
    parser.add_argument(
        '-p', '--parameter',
        help='Property used to color residues: hydrophobicity, charge, faces. Default: hydrophobicity.',
        default='hydrophobicity',
        choices=['hydrophobicity', 'charge', 'faces'],
    )
    parser.add_argument(
        '-nl', '--noletters',
        help='Show residue noletters and indices on the plot.',
        action='store_true',
    )
    parser.add_argument(
        '-hsc', '--h-scale',
        help='Hydrophobicity scale to be used. Default: eisenberg.',
        default='eisenberg',
    )
    parser.add_argument(
        '-avh', '--av-hm',
        help='Average hydrophobic moment as done by HeliQuest(R). Default: False.',
        action='store_true',
    )
    parser.add_argument(
        '-phm', '--print-hm',
        help='Show hydrophobic moment value in the title.',
        action='store_true',
    )
    parser.add_argument(
        '-phi', '--print-hi',
        help='Show hydrophobic index value in the title.',
        action='store_true',
    )
    parser.add_argument(
        '-pch', '--print-charge',
        help='Show peptide net charge in the title.',
        action='store_true',
    )
    parser.add_argument(
        '-d2', '--two-d',
        help='2D helical wheel projection (default if --three-d is not set).',
        action='store_true',
    )
    parser.add_argument(
        '-d3', '--three-d',
        help='3D helix plot.',
        action='store_true',
    )
    parser.add_argument(
        '-m', '--cmap',
        help='Matplotlib colormap name. Default: viridis.',
        default='viridis',
        type=str,
    )
    parser.add_argument(
        '-rb', '--remove-background',
        help='Remove figure background and axes.',
        action='store_true',
    )
    parser.add_argument(
        '-sv', '--save',
        help='Save the figure to a file.',
        action='store_true',
    )
    parser.add_argument(
        '-o', '--out',
        help='Output filename when --save is used. Default: output.png.',
        default='output.png',
        type=str,
    )

    vis_group = parser.add_mutually_exclusive_group()
    vis_group.add_argument(
        '-f', '--show-slices',
        help='Show slice lines/planes. Use with --slice-angle or --parameter faces.',
        action='store_true',
    )
    vis_group.add_argument(
        '-ss', '--show-sections',
        help='Show equal angular sections. Use with --sections to set section count.',
        action='store_true',
    )
    parser.add_argument(
        '-a', '--slice-angle',
        help='Slice angle in degrees. Only valid with --show-slices or --parameter faces. Default: 180.',
        default=180.0,
        type=float,
    )
    parser.add_argument(
        '-ns', '--sections',
        help='Number of angular sections. Only valid with --show-sections. Default: 18.',
        default=18,
        type=int,
    )
    parser.add_argument(
        '-nogui', '--noshow',
        help="Do not show plot. Use it together with --save",
        action='store_true'
    )

    args = parser.parse_args()

    if not args.three_d:
        args.two_d = True

    _validate_args(args, parser=parser)
    return args


# --- Entry point -----------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    args = get_arguments()
    seq_data = _run_pipeline(args)

    # Define text formater
    formater = lambda x: f"<{x}>" if args.av_hm else f"{x}"

    # print information
    length = len(args.sequence)
    print(f"Sequence length: {length} aa\n")
    aa_count = ""
    i = 0
    while i < len(args.sequence):
        new_count = f"{i+1}" if i % 10 == 0 else ' '
        aa_count += new_count
        i += len(new_count)
    print(aa_count)
    print(args.sequence, end='\n\n')

    # show radial representation
    from radial_sequence import get_radial
    radial = get_radial(seq=args.sequence, h_scale=args.h_scale)
    print("Radial representation of the core (18 residues)")
    print(f"{radial}\n")

    # show properties
    print("Basic properties ---")
    print(f"Positive residues: {seq_data['npos']}\nNegative residues: {seq_data['nneg']}")
    print(f"Net charge = {seq_data['charge']}")
    print(f"{formater('Hm')} = {seq_data['hm']}")
    print(f"{formater('Hi')} = {seq_data['hi']}")

    if not args.save:
        print("\nPlot was not saved (use --save)")

if __name__ == '__main__':
    main()
    
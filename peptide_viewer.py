# === peptide_viewer.py ===
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from sequence import Sequence
from intervals import CircleInterval

# --- Helix geometry constants ---
_THETA_DEG: float = 100.0   # rotation per residue in degrees
_INCREMENT: float = -1.0     # height increment per residue


# --- Geometry functions ----------------------------------------------------

def compute_helix_positions(seq: Sequence) -> np.ndarray:
    """
    Computes 3D helix positions for each residue using fixed helix parameters.
    Centers the helix at (0, 0, 0) by subtracting the mean position.
    Returns an array of shape (n_residues, 3).
    """
    theta_rad = np.deg2rad(_THETA_DEG)
    positions = np.array([
        [
            np.cos(i * theta_rad),
            np.sin(i * theta_rad),
            i * _INCREMENT,
        ]
        for i in range(len(seq))
    ])
    positions -= positions.mean(axis=0)
    return positions


def compute_hm_vector(seq: Sequence, positions: np.ndarray) -> np.ndarray:
    """
    Computes the hydrophobic moment vector from residue hydrophobicities
    and their XY positions. Returns a normalized 2D vector [x, y].
    """
    hm = np.zeros(2)
    for res, pos in zip(seq.residues, positions):
        hm += res.hydrophobicity * pos[:2]
    norm = np.linalg.norm(hm)
    if norm == 0:
        return np.array([1.0, 0.0])
    return hm / norm


def compute_hm_scalar(seq: Sequence, positions: np.ndarray) -> float:
    """
    Computes the hydrophobic moment scalar magnitude before alignment.
    Must be called before align_to_minus_y() to preserve the original vector.
    """
    hm = np.zeros(2)
    for res, pos in zip(seq.residues, positions):
        hm += res.hydrophobicity * pos[:2]
    return float(np.linalg.norm(hm))


def align_to_minus_y(positions: np.ndarray, hm_vector: np.ndarray) -> np.ndarray:
    """
    Rotates all positions in the XY plane so that the hydrophobic moment
    vector points toward -Y. Z coordinates are preserved unchanged.
    """
    target = np.array([0.0, -1.0])
    hx, hy = hm_vector
    tx, ty = target
    angle = np.arctan2(ty, tx) - np.arctan2(hy, hx)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotated = positions.copy()
    rotated[:, 0] = positions[:, 0] * cos_a - positions[:, 1] * sin_a
    rotated[:, 1] = positions[:, 0] * sin_a + positions[:, 1] * cos_a
    return rotated

def get_faces(
    positions: np.ndarray,
    seq: Sequence,
    phi_deg: float,
) -> tuple:
    """
    Splits residues into two faces based on their proximity to the hydrophobic
    moment vector direction (-Y after alignment).
    phi_deg is the total slice angle in degrees; phi/2 is applied on each side
    of the HM vector.
    Returns (positive_face, negative_face) as lists of residue indices.
    Reusable by GenMethod subclasses that work with helix geometry.
    """
    border = np.cos(np.deg2rad(phi_deg / 2))
    positive_face = [res.index for res, pos in zip(seq.residues, positions) if -pos[1] >= border]
    negative_face = [res.index for res, pos in zip(seq.residues, positions) if -pos[1] < border]
    return positive_face, negative_face


def compute_sections(n: int) -> list:
    """
    Divides the full circle (2pi) into n equal CircleInterval sections.
    Section 0 has its bisector pointing toward -Y (-pi/2).
    Sections are numbered counterclockwise.
    Returns a list of CircleInterval objects.
    """
    segment_angle = 2 * math.pi / n
    bisector_0 = -math.pi / 2
    section_start = bisector_0 - segment_angle / 2
    intervals = []
    for i in range(n):
        start = section_start + i * segment_angle
        end = start + segment_angle
        intervals.append(
            CircleInterval(start=start, end=end, lclosed=False, rclosed=True)
        )
    return intervals


def int_to_roman(n: int) -> str:
    """Converts a positive integer to its Roman numeral representation."""
    values  = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    symbols = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    result = ''
    for v, s in zip(values, symbols):
        while n >= v:
            result += s
            n -= v
    return result


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
        p_face, _ = get_faces(positions, seq, args.slice_angle)
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
        ax.plot([0, 0], [0, -1.5], lw=2, c='deepskyblue', zorder=1)
    else:
        ax.plot([0, 0], [0, -1.5], [0, 0], lw=2, c='deepskyblue')

    visited_xy = []
    for i, (res, pos) in enumerate(zip(seq.residues, positions)):
        x, y, z = pos
        df = 1.0

        if args.two_d:
            overlap = [np.linalg.norm(np.array([x, y]) - v) < 0.05 for v in visited_xy]
            if visited_xy and any(overlap):
                df = 1.2
            ax.scatter(x * df, y * df, c=[colors[res.index]], s=100, alpha=1, zorder=3)
            visited_xy.append(np.array([x, y]))
        else:
            ax.scatter(x, y, z, c=[colors[res.index]], s=100, alpha=0.8)

        if args.letters:
            scale = 1.4
            lx, ly = x * scale * df, y * scale * df
            label = res.letter + str(res.index + 1)
            if args.two_d:
                ax.text(lx, ly, label, size=9, ha='center', va='center')
            else:
                ax.text(lx, ly, z, label, size=9, ha='center', va='center')

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
    label_radius = 0.6

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
    ax,
) -> None:
    """Builds and sets the plot title based on active display options."""
    title = r'$\alpha$-Helix'
    extras = []
    if args.print_hm:
        extras.append(f'Hm: {round(hm_scalar, 3)}')
    if args.print_hi:
        extras.append(f'Hi: {seq.hydrophobic_index}')
    if extras:
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

def _run_pipeline(args: argparse.Namespace) -> None:
    """
    Executes the full visualization pipeline given a populated Namespace.
    Shared by both main() and plot_sequence().
    """
    seq = Sequence(args.sequence) if isinstance(args.sequence, str) else args.sequence
    cmap = cm.get_cmap(args.cmap)

    # Compute geometry before alignment to preserve the HM scalar magnitude
    positions = compute_helix_positions(seq)
    hm_vector = compute_hm_vector(seq, positions)
    hm_scalar = compute_hm_scalar(seq, positions)
    positions = align_to_minus_y(positions, hm_vector)

    colors = set_colors(args, seq, positions, cmap)
    fig, ax = create_figure(args, positions)
    set_title(args, seq, hm_scalar, ax)
    fig, ax = plot_peptide(args, seq, positions, colors, fig, ax)

    if args.show_slices:
        fig, ax = plot_slices(args, positions, fig, ax)
    elif args.show_sections:
        fig, ax = plot_sections(args, positions, fig, ax)

    fig, ax = set_legends(args, seq, positions, cmap, fig, ax)

    if args.save:
        plt.savefig(args.out, dpi=300)

    plt.show()


# --- Public API for notebook / import usage --------------------------------

def plot_sequence(
    sequence: "str | Sequence",
    parameter: str = 'hydrophobicity',
    letters: bool = True,
    print_hm: bool = False,
    print_hi: bool = False,
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
) -> None:
    """
    Visualizes a peptide as an alpha-helix projection.
    Accepts a sequence string or a Sequence object.
    All parameters mirror the CLI arguments and can be freely set from a notebook.

    Parameters
    ----------
    sequence         : Peptide sequence as string or Sequence object.
    parameter        : Coloring scheme: 'hydrophobicity', 'charge', or 'faces'.
    letters          : Show residue letters and indices on the plot.
    print_hm         : Show hydrophobic moment value in the title.
    print_hi         : Show hydrophobic index value in the title.
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
        letters=letters,
        print_hm=print_hm,
        print_hi=print_hi,
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
        '-l', '--letters',
        help='Show residue letters and indices on the plot.',
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

    args = parser.parse_args()

    if not args.three_d:
        args.two_d = True

    _validate_args(args, parser=parser)
    return args


# --- Entry point -----------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    args = get_arguments()
    _run_pipeline(args)


if __name__ == '__main__':
    main()
    
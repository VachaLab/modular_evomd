# --- sequence_logo.py ---
"""
This script contains tools to plot a sequence logo
from an evomd csv report.
"""

from collections import Counter
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties
from sequence import Sequence


def normalize_array(arr: list[float], min_target: float = 0.05, max_target:float = 1.0):
    """
    Normalizes an array to get the max = max_target and min = min_target.

    Parameters:
    - arr: Array of floats or list.
    - min_target: minimum value after normalization.
    - max_target: maximum value after normalization.
    """
    if not arr:
        return []
    
    # Get the original max and min
    max_val = max(arr)
    min_val = min(arr)
    rango_origen = max_val - min_val  # original range

    # returns max if all the elements are equal.
    if rango_origen == 0:
        return [max_target] * len(arr)

    # target range
    rango_destino = max_target - min_target

    # return normalized array
    #
    #              (x - min_val) * rango_destino
    # min_target + -----------------------------
    #                      rango_origen
    # 
    return [min_target + ((x - min_val) * rango_destino / rango_origen) for x in arr]


def plot_sequence_logo(
        sequences: list[(Sequence | str)], color: str = 'steelblue', 
        outline: bool = True, save: bool = False, out: str = 'sequence_logo.png',
        gradient: bool = True, font: (str | None) = None,
        exclude: (str|list|None) = None,
        ) -> None:
    """
    Generates a sequence logo from a list of Sequence or list of str. All the elements
    of sequence list must have the same lenght. 
    
    Parameters:
    - sequences: List of Sequence or str.
    - color: Letters color.
    - outline: Activate or deactivate letter oultine
    - save: Save figure in a file.
    - out: name of the output file.
    - gradient: whether include gradient proportional to the frequency or not.
    - font: full path for a font ttf file.
    - exclude: List of str with letters to be excluded. It helps when some sequences contain no std aa or -.
    """
    if not sequences:
        raise RuntimeError("Sequences list is empty!")

    # check lenght and transform everything into str
    sequences = [str(k) for k in sequences]
    all_len = [len(k) for k in sequences]
    seq_len = int(sum(all_len)/len(all_len))
    if seq_len != int(all_len[0]):
        raise RuntimeError("Check the sequences list. At least one sequence has a different lenght!")

    # format exclude list
    if exclude is not None:
        if isinstance(exclude, str):
            exclude = [exclude]
    else:
        exclude = []

    # check font
    if font is None:
        _SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
        funente_path = os.path.join(_SCRIPTDIR, "fonts", "SilkRemington-SBold.ttf")
    else:
        funente_path = font
    
    # Compute relative frequency per position
    freqs_per_pos = []
    for i in range(seq_len):
        columna = [seq[i] for seq in sequences if seq[i] not in exclude]
        conteos = Counter(columna)
        # frecuencias is a dictionary 
        frecuencias = {aa: count / len(columna) for aa, count in conteos.items()}
        
        # reverse=True places the highest freq at the beginning
        frecuencias_ordenadas = sorted(frecuencias.items(), key=lambda x: x[1], reverse=True)
        freqs_per_pos.append(frecuencias_ordenadas)

    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, seq_len * 0.5), 4))
    
    # Use the font in fonts/ directory
    propiedades_fuente = FontProperties(fname=funente_path)
    
    # Compute the width for each letter
    letras_posibles = "ACDEFGHIKLMNPQRSTVWY"
    anchos = []  # widths
    for letra in letras_posibles:
        tp = TextPath((0, 0), letra, size=1, prop=propiedades_fuente)
        anchos.append(tp.get_extents().width)
    max_width = max(anchos) if anchos else 1.0
    
    # Scale factor (0.8 per column)
    scale_x_global = 0.8 / max_width
    
    # Contour configuration
    edge_color = 'black' if outline else 'none'
    line_width = 0.5 if outline else 0

    # 4. Dibujar las letras
    for i, pos_freqs in enumerate(freqs_per_pos):
        x_pos = i + 1
        y_offset = 0.0

        if gradient:
            alphas = normalize_array([k[1] for k in pos_freqs])
        else:
            alphas = [1 for k in pos_freqs]

        a = 0
        for aa, freq in pos_freqs:
            if freq == 0: 
                continue
            
            tp = TextPath((0, 0), aa, size=1, prop=propiedades_fuente)
            bbox = tp.get_extents()
            
            if bbox.width == 0 or bbox.height == 0:
                continue
            
            # Y scale depends on relative frequency
            scale_y = freq / bbox.height
            
            # Center letter
            width_escalado = bbox.width * scale_x_global
            posicion_x_centro = x_pos - (width_escalado / 2)
            
            transformacion = Affine2D() \
                .scale(scale_x_global, scale_y) \
                .translate(posicion_x_centro - (bbox.x0 * scale_x_global), y_offset - (bbox.y0 * scale_y))
            
            tp_transformado = tp.transformed(transformacion)
            
            # Add patches
            patch = PathPatch(tp_transformado, facecolor=color, edgecolor=edge_color, linewidth=line_width, alpha=alphas[a])
            ax.add_patch(patch)
            
            y_offset += freq
            a += 1

    # Axes
    ax.set_xlim(0.5, seq_len + 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(1, seq_len + 1))
    
    ax.set_ylabel('Relative frequency')
    ax.set_xlabel('Position')
    
    # Clean borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save:
        plt.savefig(out, dpi=300)
    plt.show()

# --- Argument parsing ------------------------------------------------------
def get_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the sequence logo creator."""
    parser = argparse.ArgumentParser(
        description='Sequence logo creator from csv.'
    )
    parser.add_argument(
        '-rep', '--report',
        help='CSV report from evomd.',
        default=None,
    )
    parser.add_argument(
        '-evopkl', '--evopkl',
        help='Binary (pickle) Evolver file previously created. Ignored if --report is given. Default: evolver.pkl',
        default='evolver.pkl'
    )
    parser.add_argument(
            '-r', '--ratio',
            help='Ratio of sequences used to create the sequence logo. Default: 1.0',
            type=float,
            default=1.0,
        )
    parser.add_argument(
            '-gp', '--group',
            help='Set group of sequences to be used. Default: max. Only useful if --ratio < 1.0',
            default='max',
            type=str.lower,
            choices=['max', 'min'],
        )
    parser.add_argument(
            '-i', '--ignore-fitness',
            help='Ignore the fitness and plot all the sequences. This ignores --group and --ratio.',
            action='store_true',
        )
    parser.add_argument(
        '-g', '--gradient',
        help='Plot with a alpha-gradient proportional to the relative frequency.',
        action='store_true'
    )
    parser.add_argument(
        '-s', '--save',
        help='Save the sequence logo as image.',
        action='store_true'
    )
    parser.add_argument(
        '-o', '--out',
        help='Name of the output image. Used if --save is True.',
        default='sequence_logo.png',
    )
    
    args = parser.parse_args()

    # ignore evopkl if report is not None
    if args.report is not None:
        args.evopkl = None

    # ignore group and ratio if ignore_fitness is True
    if args.ignore_fitness:
        args.group = None
        args.ratio = None

    # reverse is used for sorting sequences
    args.reverse = (args.group == 'max')

    return args

def from_report(args: argparse.Namespace) -> list:
    import csv
    import math
    sequences = []

    with open(args.report, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_seq = (row.get('sequence') or '').strip()
            raw_fit = (row.get('fitness') or '').strip()

            if not raw_seq:
                continue

            # discard rows without a valid fitness
            if not args.ignore_fitness:
                if raw_fit == '' or raw_fit.lower() == 'none':
                    print(f'Evolver.read_report: discarding {raw_seq} (no fitness)')
                    continue
                try:
                    fitness = float(raw_fit)
                except ValueError:
                    print(f'Evolver.read_report: invalid fitness "{raw_fit}" for {raw_seq} --> discarding')
                    continue
                if math.isnan(fitness):
                    print(f'Evolver.read_report: discarding {raw_seq} (nan fitness)')
                    continue

            # build Sequence; fitness is a read-only property, set it via fitness_list
            new_seq = Sequence(raw_seq)
            new_seq.fitness_list = [fitness]
            sequences.append(new_seq)

    return sequences

def from_evoplk(args: argparse.Namespace) -> list:
    from evolver import Evolver
    import utils
    evo_pre = args.evopkl
    if not utils.exists(evo_pre):
        raise RuntimeError(f"Evolver not found: {evo_pre}")
    # Fall back to the default pickle in the working directory
    evo = utils.read_pkl(evo_pre)
    sequences = evo.parent_sequences + evo.discarded_sequences + evo.sequences
    if not args.ignore_fitness:
        sequences = [k for k in sequences if len(k.fitness_list)>0]
    return sequences

def main():
    """
    Function executed as main script.
    """
    args = get_arguments()

    if args.report:
        sequences = from_report(args)
    else:
        sequences = from_evoplk(args)

    if len(sequences) == 0:
        raise RuntimeError("No sequences for logo.")

    # sort sequences if not ignore_fitness
    if not args.ignore_fitness:
        sequences.sort(key=lambda x: x.fitness, reverse=args.reverse)
        # select the ratio
        amount = int(len(sequences)*args.ratio)
        sequences = sequences[:amount]

    plot_sequence_logo(sequences=sequences, gradient=args.gradient, save=args.save, out=args.out)

if __name__ == '__main__':
    main()

# === peptide_alignment.py ===
import yaml
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sequencearray import SequenceArray
from sequence import Sequence


def get_arguments() -> argparse.Namespace:
    """
    Peptide viewer arguments from command-line
    """
    parser = argparse.ArgumentParser(description='Peptide 3D-viewer :)')

    # Input file with sequences and additional options
    parser.add_argument(
        '-f', '--file',
        help='yaml file with list of sequences',
        default=None
    )
    parser.add_argument(
        '-hs', '--h-scale',
        help='Hydrophobicity scale (default: eisenberg)',
        default='eisenberg'
    )
    parser.add_argument(
        '-ri', '--read-input',
        help='Ignore file and read sequences from iterative command-line input',
        action='store_true',
    )

    args = parser.parse_args()

    return args

# read yaml
def load_yaml(filename) -> dict:
    with open(filename, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader) or {}    

# iterative input
def input_sequences() -> list:
    print('Enter sequences one by one. Enter an empty line to finish input.')
    sequences = []
    i = 0
    while True:
        candidate = input(f'Sequence {i}: ')
        if len(candidate) == 0:
            break
        sequences.append(candidate)
        i += 1
    return sequences


def main():
    args = get_arguments()
    if args.read_input or not args.file:
        sequences = input_sequences()
    else:
        yaml_data = load_yaml(args.file)
        sequences = yaml_data['sequences']

    sequences = SequenceArray(sequences=[Sequence(k) for k in sequences])
    print(sequences)


if __name__ == '__main__':
    main()


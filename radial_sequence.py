# === radial_sequence.py ===
import math
import sys
import numpy as np
from sequence import Sequence
from sequence_geometry import *
from intervals import CircleInterval

_N_SECTIONS: int = 18


def _compute_exclude(seq_len: int) -> list[int]:
    """
    Computes the list of residue indices to discard automatically so that
    the core contains at most _N_SECTIONS residues.
    Discards symmetrically from both termini; any odd remainder is removed
    from the C-terminal end.
    Returns an empty list if seq_len <= _N_SECTIONS.
    """
    n_discard = seq_len - _N_SECTIONS
    if n_discard <= 0:
        return []
    n_left = n_discard // 2
    n_right = n_discard - n_left  # absorbs the odd remainder at C-terminus
    left_indices = list(range(n_left))
    right_indices = list(range(seq_len - n_right, seq_len))
    return left_indices + right_indices


def _angular_distance(a: float, b: float) -> float:
    """
    Computes the shortest angular distance between two angles in radians.
    Result is always in [0, pi].
    """
    diff = abs(a - b) % (2 * math.pi)
    return min(diff, 2 * math.pi - diff)

def _get_res_sec(sec: int, residues: list) -> str:
    """
    Get the letter of a residue or gives '-' if no residue belongs
    to section sec.
    """
    for res, val in residues:
        if val == sec:
            return res
    return '-'


def get_radial(seq: Sequence | str, exclude: list[int] | None = None) -> str:
    """
    Converts a primary peptide sequence into an 18-position radial sequence
    based on the alpha-helix projection aligned to the hydrophobic moment.

    Each of the 18 angular sections of the helical wheel receives the letter
    of the residue whose XY angle is closest to that section's bisector.
    Sections without a residue are represented by '-'.

    Parameters
    ----------
    seq     : Peptide sequence as a string or Sequence object.
    exclude : Optional list of residue indices (0-based) to discard before
              computing the core. If None, the core is selected automatically
              by trimming residues symmetrically from both termini, with any
              odd remainder removed from the C-terminal end.

    Returns
    -------
    str
        An 18-character string representing the radial sequence, where each
        character corresponds to one angular section (I to XVIII) and '-'
        marks empty sections.
    """
    # Normalize input to Sequence
    if not isinstance(seq, Sequence):
        seq = Sequence(seq)

    # Compute helix geometry on the full sequence before trimming,
    # so that the hydrophobic moment reflects the complete peptide.
    positions = compute_helix_positions(seq)
    hm_vector = compute_hm_vector(seq, positions)
    positions = align_to(positions, hm_vector, target=np.array([1., 0.]))

    # Determine indices to exclude
    if exclude is None:
        exclude_set = set(_compute_exclude(len(seq)))
    else:
        exclude_set = set(exclude)

    # Build core: (residue, aligned_position) pairs for non-excluded residues
    core = [
        (res, positions[res.index])
        for res in seq.residues
        if res.index not in exclude_set
    ]

    # Compute section angle
    segment_angle = 2 * math.pi / _N_SECTIONS

    # Create intervals for sections
    segment_start = 0 - segment_angle / 2
    intervals = []
    for n, sg in enumerate(range(_N_SECTIONS)):
        start = segment_start + sg * segment_angle
        end = segment_angle + start
        intervals.append(CircleInterval(start=start, end=end, lclosed=False, rclosed=True, tag=n))

    print(len(intervals))
    print(intervals)
    # Assign each core residue to the section
    # Find section by lambda function
    find_section = lambda a: [interval.tag for interval in intervals if interval(a)][0]

    section_map: list[int, str] = []  # section_index -> residue letter
    for res, pos in core:
        angle = np.mod(math.atan2(pos[1], pos[0]), 2*np.pi)
        sec = find_section(angle)
        section_map.append([res.letter, sec])
    
    # Sort residues by section
    section_map.sort(key=lambda k: k[1])
    print(section_map)
    # Build the radial sequence from section I (index 0) to XVIII (index 17)
    radial = []
    for n in range(_N_SECTIONS):
        rep = _get_res_sec(n, section_map)
        radial.append(rep)
    radial = ''.join(radial)

    return radial


def main() -> None:
    sequence = sys.argv[1]
    radial = get_radial(seq=sequence)
    print(radial)


if __name__ == '__main__':
    main()
    
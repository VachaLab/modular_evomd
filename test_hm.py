# === test_hm.py ===
"""
Example external-methods module for Evo-MD.
This example can be used with `input.yaml` input file
in test directory. It performs the maximization of hydrophobic moment.

This is a minimal example of the user-supplied modules that
Evo-MD calls each iteration. It is NOT part of the core project.

A module like this provides the four functions the Manager looks up by name:
    constructor_method(sequence) -> None
    calculator_method(sequence)  -> None
    calculator_check(sequence)   -> bool
    analyzer_method(sequence)    -> float

You point the Instructor at this module through the YAML keys `constructor`,
`calculator`, `calculator_check` and `analyzer` (here all four are 'test_hm'). 
    constructor: test_hm
    calculator:  test_hm
    calculator_check:  test_hm
    analyzer:    test_hm

The four functions can be defined in the same module and called as
`calculator` key.

Each function receives the Sequence object as an argument `sequence`
and runs in sequence's iteration directory (iter_?). In this example the
"simulation" is faked: there is nothing to build or submit, the check pretends
the job needs one extra cycle, and the fitness is simply the peptide's
hydrophobic moment. 
"""

import logging

logger = logging.getLogger(__name__)


def constructor_method(sequence) -> None:
    """
    Prepare the simulation system for one sequence.

    Called first each iteration, inside the sequence's directory. A real
    implementation would write input files / build the system here. This example
    has nothing to build, so it just prints.
    """
    print(f'Nothing to create for {sequence}')


def calculator_method(sequence) -> None:
    """
    Launch the calculation for one sequence.

    Called after the constructor. A real implementation would submit the job to
    external software. This example only prints the sequence.
    """
    print('Check sequence')
    print(sequence)


def calculator_check(sequence) -> bool:
    """
    Report whether the calculation for one sequence has finished.

    Polled repeatedly by the Manager until it returns True (or the cycle budget
    runs out). Must return a bool.

    This example fakes a job that needs one extra poll: the first call attaches
    a `check_times` marker to the sequence and returns False; the next call sees
    the marker and returns True. 
    """
    try:
        # second poll onward: the marker exists --> finished
        times = sequence.check_times
        if times > 0:
            return True
    except:
        # first poll: set the marker and report not-yet-ready
        sequence.check_times = 1
        return False
    # fallback: not ready
    return False


def analyzer_method(sequence) -> float:
    """
    Compute and return the fitness for one finished sequence.

    Called once the check reports the job ready; the returned float is appended
    to the sequence's fitness history. Runs inside the sequence's directory, so
    the file written here lands there.

    This example uses the peptide's hydrophobic moment as the fitness and also
    dumps a small fitness.txt for inspection.
    """
    from sequence_geometry import compute_hm_scalar, compute_helix_positions
    positions = compute_helix_positions(sequence)
    hm = compute_hm_scalar(sequence, positions)
    with open('fitness.txt', 'w') as f:
        f.write(f'seq: {sequence}\nhm: {hm}\ncharge: {sequence.charge}\n{sequence.hydrophobic_scale}\n')
    return hm


if __name__ == '__main__':
    pass

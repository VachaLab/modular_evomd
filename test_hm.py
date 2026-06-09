# === test_hm.py ===
"""
Example external-methods module for Evo-MD.

This is a minimal, self-contained example of the user-supplied module that
Evo-MD calls each iteration. It is NOT part of the core project; it exists to
show how to wire your own simulation/fitness code into the framework.

A module like this provides the four functions the Manager looks up by name:
    constructor_method(sequence) -> None
    calculator_method(sequence)  -> None
    calculator_check(sequence)   -> bool
    analyzer_method(sequence)    -> float

You point the Instructor at this module through the YAML keys `constructor`,
`calculator`, and `analyzer` (here all three are 'test_hm'). The same module may
provide all four functions, as it does below.

Each function receives the Sequence object as the keyword argument `sequence`
and runs inside that sequence's iteration directory. In this example the
"simulation" is faked: there is nothing to build or submit, the check pretends
the job needs one extra cycle, and the fitness is simply the peptide's
hydrophobic moment. Replace the bodies with real calls (build inputs, submit
jobs, parse results) to use Evo-MD for an actual problem.
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
    the marker and returns True. The first try/except also guards against a
    sequence that does not expose hydrophobic_index.
    """
    try:
        value = sequence.hydrophobic_index
        print(f'Sequence is ready: {value}')
    except:
        # sequence object is not what we expected --> not ready
        print('Where is the sequence?')
        return False
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
        f.write(f'seq: {sequence}\nhm: {hm}\ncharge: {sequence.charge}\n')
    return hm


if __name__ == '__main__':
    pass

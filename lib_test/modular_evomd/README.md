# evomd

A molecular-simulation-driven evolutionary engine for peptide (amino acid)
sequence optimization. Usable both as a set of command-line tools and as an
importable Python library.

## Installation

```bash
pip install git+https://github.com/VachaLab/modular_evomd.git
```

This installs the `evomd` library and three console commands: `evo-md`,
`peptide-viewer`, and `sequence-logo`.

## Command-line usage

```bash
# Create a new session from a YAML instruction file and inspect it
evo-md -f instructions.yaml --create-evolver
evo-md --show-evolver

# Run the evolution loop
evo-md --start

# If it gets interrupted (Ctrl+C, crash, etc.), resume it
evo-md --restart

# Intentionally stop a session so a later --start/--restart doesn't touch it
# by accident (useful when juggling several evolutions/pkl files at once)
evo-md --stop-evolver

# Visualization tools
peptide-viewer --sequence KLAKLAKKLAKLAK
sequence-logo --report sequences_report.csv
```

Run `evo-md --help`, `peptide-viewer --help`, or `sequence-logo --help` for
the full list of options.

## Library usage

```python
import evomd

instructions = """
population: 32
peptide_len: 22
populate_method: swap
max_generations: 50
calculator: my_calculator_module
"""

evo = evomd.create_evolver(instructions)  # a YAML file path also works here
evo.show_evolver()
evo.start(fast_cycle=True)
evo.plot_evolution()
```

`create_evolver()` accepts either a path to a YAML file or the YAML content
itself as a string, so the same call works from a script or pasted directly
in a notebook cell.

If a run gets interrupted mid-session (e.g. `KeyboardInterrupt` in a
notebook), resume it on the same `evo` object with:

```python
evo.restart()
```

and stop it intentionally (equivalent to `evo-md --stop-evolver`) with:

```python
evo.stop_evolver()
```

### Other library entry points

```python
from evomd import Sequence, Residue, Scales, Instructor, Evolver, Manager, Generator
from evomd.viz.peptide_viewer import plot_sequence
from evomd.viz.sequence_logo import plot_sequence_logo
```

## External simulation methods

`evomd` does not define the fitness function or the simulation itself: it
calls into a user-supplied Python module (named via the `constructor`,
`calculator`, `calculator_check`, and `analyzer` options in the YAML
instructions), each exposing a fixed set of functions:

```python
def constructor_method(sequence): ...   # prepares the simulation input
def calculator_method(sequence): ...    # launches the simulation
def calculator_check(sequence): ...     # returns True once it has finished
def analyzer_method(sequence): ...      # returns the fitness value
```

See `examples/example_instructions.yaml` for a minimal configuration.

## Repository layout

```
src/evomd/
├── core/     configuration, the evolutionary engine, and the sequence data model
├── methods/  sequence-generation strategies (swap, hybrid, pattern, flanks, ...)
├── viz/      peptide viewer, sequence logo, radial sequence plots
└── cli/      the evo-md command-line entry point
```

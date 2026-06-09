# Modular EvoMD implementation in python

EvoMD is an evolutionary optimization framework for peptide sequences. It evolves a population of peptides based on a user-defined fitness function evaluated via simulation modules written and plugged in by user. The configuration is stored in a single YAML file, and the state is serialized to evolver.pkl after every step.

---

## Dependencies

Python 3.10+ required.

```bash
pip install numpy pyyaml matplotlib
```

---

## How it works

Each generation EvoMD:

1. **Constructs** simulation systems for the current population.
2. **Runs** your external simulation (or any fitness computation).
3. **Checks** when each simulation finishes.
4. **Analyzes** results into a fitness value per sequence.
5. **Sorts** sequences, keeps the best as parents, and generates the next generation.

User controls what happens in steps 1–4 by writing a Python plug-in with four functions and calling it in the YAML. EvoMD handles the rest.

The general idea of the modular architecture is shown in the next figure:

![EvoMD architecture](images/evomd_architecture.png)

---

## Quickstart

This example uses `test_hm.py` (included), which maximizes the hydrophobic moment of 20-residue peptides with no real simulation.

**1. Save this as `input.yaml`:**

```yaml
optimize: maximize
population: 32
peptide_len: 20
populate_method: swap
extra_mutation: True
also_mutate_probability: 0.1
parents_ratio: 0.25

constructor: test_hm
calculator:  test_hm
analyzer:    test_hm

sleep_time: 0
max_check_cycle: 50
max_generations: 50

hindex_restriction: True
hindex_min: -8
hdistribution_restriction: True
hdistribution_threshold: 0.9
```

**2. Create the Evolver:**

```bash
python evo-md.py --file input.yaml --create-evolver
```

**3. Run the evolution:**

```bash
python evo-md.py --file input.yaml --start
```

**4. Inspect the results:**

```bash
python evo-md.py --show-evolver
python evo-md.py --report-sequences            # writes sequences_report.csv
python evo-md.py --plot-evolution --show-kids  # plots and saves as evolution.png
```

---

## Writing your own methods

Users must create Python modules with these four functions and name it in the YAML (`constructor`, `calculator`, `analyzer`). All three keys can point to the same file.
`calculator_method` and `calculator_check` must be in the same python file.

```python
def constructor_method(sequence) -> None:
    # Write input files, build the system, etc.
    pass

def calculator_method(sequence) -> None:
    # Launch the job (submit, start a process, etc.)
    pass

def calculator_check(sequence) -> bool:
    # Return True when the job is done, False otherwise.
    # Polled repeatedly by EvoMD until True or the cycle budget runs out.
    return True

def analyzer_method(sequence) -> float:
    # Parse results and return the fitness value.
    return 0.0
```

Each function runs inside the sequence's iteration directory. The `sequence` object gives the residue string (`str(sequence)`), physicochemical properties (`sequence.charge`, `sequence.residues`, …), and state flags (`sequence.is_failed` to mark a failure). You can attach arbitrary attributes to carry information between functions (e.g. a job ID set in `calculator_method` and read in `calculator_check`).

See `test_hm.py` for a short example.

---

## Common commands

| What you want to do | Command |
|---|---|
| Create a new Evolver and exit | `python evo-md.py --file input.yaml --create-evolver` |
| Start the evolution loop | `python evo-md.py --file input.yaml --start` |
| Resume an interrupted iteration | `python evo-md.py --restart` |
| Stop after the current iteration | `python evo-md.py --stop-evolver` |
| Show the current state | `python evo-md.py --show-evolver` |
| Export all sequences to CSV | `python evo-md.py --report-sequences` |
| Plot fitness over generations | `python evo-md.py --plot-evolution` |
| Roll back to the last complete generation | `python evo-md.py --last-generation` |
| Show the loaded configuration | `python evo-md.py --show-current` |

Run `python evo-md.py --help` for the full flag reference.

---

## Output files

| File | Description |
|---|---|
| `evolver.pkl` | Full Evolver state, updated after each step. |
| `sequences_report.csv` | All sequences with generation and fitness. |
| `evolution.png` | Fitness plot (written by `--plot-evolution`). |
| `simulation_data/<SEQUENCE>/iter_N/` | Per-sequence, per-attempt directories where your methods run. |

---

## Recovery

If a run is interrupted:

```bash
python evo-md.py --restart
```

If the last generation is incomplete or just want to go back to the last completed generation:

```bash
python evo-md.py --last-generation
```

Then continue with `--start`.

---


# Modular EvoMD

A molecular-simulation-driven evolutionary engine for peptide
sequence optimization. Usable both as a set of command-line tools and as an
importable Python library.

---

## Installation

```bash
pip install git+https://github.com/VachaLab/modular_evomd.git@evolib
```

This installs the `evomd` library and three console commands: `evo-md`,
`peptide-viewer`, and `sequence-logo`.

---

## Dependencies

Python 3.10+ required.

Other libraries used:

- numpy
- matplotlib
- pyyaml


---

## Quickstart

This example uses `test_hm.py` (found in examples directory), which maximizes the hydrophobic moment with no real simulation.

**1. Create an `input.yaml` (or use file in examples directory):**

```yaml
optimize: maximize  # maximize or minimize
population: 32  # how many sequences in the population?
peptide_len: 20  # peptide lenght
populate_method: swap  # method for creating child sequences: hybrids or swap work well
extra_mutation: True  # Include a random point mutation?
also_mutate_probability: 0.1  # 10% of the new sequences are mutated
parents_ratio: 0.25  # Parents are chosen from the 25% of the population
include_parents: False # If True, parent sequences are simulated again

# User must write the name of the python script with the external methods.
# These are external methods. You can include all the methods in one file
# and define only calculator (see example in examples/input.yaml).
constructor: test_hm  # script for contructing simulation box.
calculator:  test_hm  # running.
calculator_check: test_hm  # checking simulations.
analyzer:    test_hm  # Analysis and computation of fitness values.

sleep_time: 0  # check if simulations finished every sleep_time seconds
max_check_cycle: 50  # maximum cycles 
max_generations: 30  # Maximum number of generations

hydrophobic_restriction: False  # restraints in hydrophobic moment
hydrophobic_min: null   # ignored while hydrophobic_restriction is False
hydrophobic_max: null   # no max limit

mut_aa: ADEFKLNQRSTY    # only these aa are used to create sequences
```

**2. Create the Evolver:**

Use the command evo-md to create an `evolver.pkl` file.

```bash
evo-md --file input.yaml --create-evolver
```

**3. Run the evolution:**

```bash
evo-md --start
```

**4. Inspect the results:**

```bash
evo-md --show-evolver                 # prints the final state of the evolution process
evo-md --report-sequences             # writes sequences_report.csv
evo-md --plot-evolution --show-kids   # plots and saves as evolution.png
```

The plot shows the evolution of the population (black solid line). Green solid line is the average fitness of the kids
created in each generation. Highest and lowest fitness found in each generation are also presented.

<div style="text-align:center;">
    <img src="images/evolution_example.png" width="400" alt="Evolution">
</div>

**5. Plot sequences:**

You can use `peptide-viewer` to plot any sequence as $\alpha$-helix. You can show hydrophobic moment computed as described by [Eisenberg](https://doi.org/10.1073/pnas.81.1.140).

The example shows Opi1 peptide (`QKLSRAIAKGKDNLKEYKLNMS`).

```bash
peptide-viewer --sequence QKLSRAIAKGKDNLKEYKLNMS
```

Do you want to see information as shown by [HeliQuest](https://heliquest.ipmc.cnrs.fr)? --> Change hydrophobicity scale and include the information that you need.

```bash
peptide-viewer --sequence QKLSRAIAKGKDNLKEYKLNMS --print-hm --print-hi --print-ch --av-hm --h-scale fauchere-pliska
```

<div style="text-align:center;">
    <img src="images/opi1_example.png" width="400" alt="Peptide viewer">
</div>

**6. Plot sequence logo:**

You can use `sequence-logo` to see the behavior of the primary sequences as a sequence logo. 
`sequence-logo` can read evolver.pkl or the CSV report created by the command `evo-md --report-sequences`.

```bash
# plot the 25% of the sequences with highest hydrophobic moment
sequence-logo --ratio 0.25 --gradient --group max --evopkl evolver.pkl
```

<div style="text-align:center;">
    <img src="images/sequence_logo.png" width="400" alt="Peptide viewer">
</div>

**Explore the available options**

```bash
evo-md --help          # evolution
peptide-viewer --help  # helix view
sequence-logo --help   # sequence logo
```

### Library usage

There is a worked example in examples directory.
You'll need jupyter lab. 

### Running on LUMI

You can run evo-md using a LUMI container wrapper.
Follow the instructions in branch `developement`.

---

## More options

### Populating from backup

You can populate Evolver from the json files created after each generation.
First step is to **create a new evolver** using an input file with an adequate configuration 
(be sure that peptide_len is equal to the length of the sequences in the backup).

Then populate from backup:

```bash
evo-md --from-backup
```

This creates sequences from json files in simulation directory and sort the sequences. 
The result is an evolver.pkl file with choosen parents ready to start.

```bash
evo-md --start
```


### Creating an Evolver from a CSV report

You can create an evolver.pkl file from the CSV report created by the command
`evo-md --report-sequences`.

```bash
evo-md --create-evolver --read-report NAME.csv
```

You can read any CSV file, but the columns `sequence`, `generation`, and `fitness` must be 
present. This is also useful to update evo-md version.

### Update version

Use the old version to create a sequence report in CSV format.

```bash
evo-md --report-sequences
```

Update input.yaml with new configuration.

**Create a new evolver.pkl** and read report with the new version.

```bash
evo-md --create-evolver --file input.yaml --read-report sequence_report.csv
```

Start evolution with the new version

```bash
evo-md --start
```

### Loading an existing evolver.pkl

```python
evo = evomd.load_evolver("evolver.pkl")  # path defaults to "evolver.pkl"
evo.show_evolver()
evo.restart()  # continue it, if it wasn't finished
```

---

## Structure of modular EvoMD

The general process begins with a random population that is continuously evaluated and sorted. 
The worst performers are discarded, and the population is renewed through crossover of the best sequences. 
The result is an optimized population of sequences. 

<div style="text-align:center;">
    <img src="images/evomd_process.png" width="400" alt="Peptide viewer">
</div>

In each generation EvoMD:

1. **Constructs** the simulation box (constructor_method).
2. **Runs** the simulation (calculator_method).
3. **Checks** if simulations finished (calculator_check).
4. **Analyzes** results and computes fitness (analyzer_method).
5. **Sorts** sequences, keeps the best as parents, and generates the next generation.

Users have the control of steps 1–4 by writing a Python plug-in with four functions and calling it in the YAML (see next section). EvoMD handles the evolution itself.

The general idea of the modular architecture is shown in the next figure:

<div style="text-align:center;">
    <img src="images/evomd_architecture.png" width="400" alt="Peptide viewer">
</div>

Class `Evolver` contains the lists of sequences and sorts them. 
Class `Manager` helps `Evolver` to read information from external methods.

### External simulation methods

`evomd` does not define the fitness function or the simulation itself: it
calls into a user-supplied Python module (named via the `constructor`,
`calculator`, `calculator_check`, and `analyzer` options in the YAML
instructions), each exposing a fixed set of functions:


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

## Repository layout

```
src/evomd/
├── core/     configuration, the evolutionary engine, and the sequence data model
├── methods/  sequence-generation strategies (swap, hybrid, pattern, flanks, ...)
├── viz/      peptide viewer, sequence logo, radial sequence plots
└── cli/      the evo-md command-line entry point
```

---
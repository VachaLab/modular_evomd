# === instructor.py ===
"""
Instructor: configuration holder for an Evo-MD run.

This module reads a YAML instruction file and turns it into an Instructor
object whose attributes drive the evolution process. Each valid option is
declared once in the class-level `_schema` as an Instruction descriptor, which
defines its type, default value, and any validation (choices, ranges, list
subtypes). Options absent from the YAML fall back to their declared default.

The Instructor holds configuration related to the *evolution process* only
(population size, generation methods, restrictions, parent selection, stopping
criteria). It deliberately does NOT define the fitness function or the
simulation methods themselves: those live in user-supplied external modules
named via the `constructor`, `calculator`, and `analyzer` options.

Besides storing configuration, the Instructor builds the Generator: it
assembles the requested generation methods (configure via set_gen_methods),
the active restrictions (configure_restrictions), and exposes the result as
self.generator for the Evolver to use.

AH Tanguma
"""

import logging
from typing import Dict
import os
import yaml
from instruction_fields import Instruction
from generator import Generator

logger = logging.getLogger(__name__)


class Instructor:
    """
    Holds and validates the configuration for an Evo-MD run.

    Valid options are declared as Instruction descriptors in the class-level
    `_schema`. On construction, each option is read from the YAML file (or set
    to its default if absent), then the Generator is configured from the
    resulting values.

    All class attributes below are configuration options; the inline comment on
    each one documents its meaning, units, and accepted values where relevant.
    """

    name = 'instructor'
    _schema: Dict[str, Instruction] = {}
    # Each attribute below is a valid instruction, declared with its expected
    # type and default value. The Instruction descriptor registers itself in
    # _schema and enforces type/choice/range validation on assignment.

    # --- basic configuration -------------------------------------------------
    evomd_directory = Instruction(str, 'simulation_data')  # working dir for simulation data
    evolver_name = Instruction(str, 'evolver')             # base name for the Evolver and its pkl file
    optimize = Instruction(str, 'maximize', choices={'maximize', 'minimize'})  # optimization direction
    hydrofobic_scale = Instruction(str, 'eisenberg', choices={'eisenberg', 'kyte-doolittle', 'wimley-white', 'fauchere-pliska'})  # hydrophobicity scale used throughout

    # --- sequence lists ------------------------------------------------------
    sequences = Instruction(list, [], subtype=str)           # initial seed sequences (optional)
    excluded_sequences = Instruction(list, [], subtype=str)  # forbidden sequences, never allowed in the population

    # --- showing evolver -----------------------------------------------------
    top_list = Instruction(int, 10)  # number of top sequences shown by --show-evolver

    # --- population ----------------------------------------------------------
    mut_aa = Instruction(str, 'ACDEFGHIKLMNPQRSTVWY')  # residue pool for mutations (default: all 20 natural amino acids)
    peptide_len = Instruction(int, 22)                 # length of every peptide in the population
    population = Instruction(int, 120)                 # number of sequences simulated per generation
    populate_method = Instruction(
        (str, list),
        ['swap'],
        choices={'hybrids', 'group_mutation', 'hydrophobic_mutation', 'swap', 'faces', 'pattern', 'flanks'},
        choice_list=True, normalize=lambda v: [v] if isinstance(v, str) else list(v),)
    # generation method(s) used to produce children. May be a single name or a
    # list. 'pattern' must be used alone; 'flanks' wraps an inner method (see
    # set_gen_methods for the combination rules).
    method_weights = Instruction(list, None, subtype=float)  # selection weights per method (must match populate_method length; equal if omitted)
    extra_mutation = Instruction(bool, True)                 # apply an additional random mutation after the main method
    also_mutate_probability = Instruction(float, 0.1, range=[0, 1])  # per-residue mutation probability (only used when extra_mutation is True)
    include_parents = Instruction(bool, False)               # if True, parents are simulated too and re-inserted into the next generation
    avoid_reinsertion = Instruction(bool, True)              # if True, an already-tested sequence is treated as forbidden (no reinsertion)

    # --- for faces method ----------------------------------------------------
    face_slice_angle = Instruction(float, 180, range=[0, 360])  # slice angle for the 'faces' method: half-angle on each side of the hydrophobic vector

    # --- work with patterns --------------------------------------------------
    pattern = Instruction(str, None)          # template string; '-' marks positions that may be modified, other letters are fixed
    pattern_options = Instruction(list, None) # residues allowed at each free '-' position (defaults to mut_aa)
    pattern_weights = Instruction(list, None) # weights for each option (defaults to equal weights)
    pattern_maxmut = Instruction(int, 1)      # maximum number of positions mutated at once

    # --- work with flanks ----------------------------------------------------
    n_flank = Instruction(str, None)  # fixed N-terminal fragment locked by the 'flanks' method
    c_flank = Instruction(str, None)  # fixed C-terminal fragment locked by the 'flanks' method
    flank_inner = Instruction(str, 'swap', choices={'hybrids', 'group_mutation', 'hydrophobic_mutation', 'swap'})  # inner method applied to the core between flanks

    # --- choosing parents ----------------------------------------------------
    parents_ratio = Instruction(float, 0.3)        # fraction of the population kept as parents (0.3 = 30% parents, 70% discarded)
    populate_weighted = Instruction(bool, False)   # if True, better-ranked peptides are preferred when chosen as parents
    weight_bias = Instruction(float, 0.3)          # rank weighting strength: weight = (population - index) * weight_bias
    include_discarded = Instruction(bool, False)   # if True, discarded sequences may also be chosen as parents

    # --- restrictions --------------------------------------------------------
    # Each restriction below is only applied when its *_restriction flag is True
    # (or, for the count/pattern ones, when the relevant value is non-default).
    hydrophobic_restriction = Instruction(bool, True)  # enforce hydrophobic-moment bounds
    hydrophobic_min = Instruction(float, 5.5)          # minimum hydrophobic moment
    hydrophobic_max = Instruction(float, +100)         # maximum hydrophobic moment

    hindex_restriction = Instruction(bool, False)  # enforce hydrophobicity-index bounds
    hindex_min = Instruction(float, -9.0)          # minimum hydrophobicity index
    hindex_max = Instruction(float, -6.5)          # maximum hydrophobicity index

    charge_restriction = Instruction(bool, False)  # enforce net-charge bounds
    charge_min = Instruction(float, -100)          # minimum net charge
    charge_max = Instruction(float, +100)          # maximum net charge

    prohibited_patterns = Instruction(list, [], subtype=str)  # forbidden residue runs; e.g. 'KKK' forbids three or more consecutive K

    positive_atleast = Instruction(int, 0)  # require at least this many positively charged residues (0 = no requirement)
    negative_atleast = Instruction(int, 0)  # require at least this many negatively charged residues (0 = no requirement)

    hdistribution_restriction = Instruction(bool, False)  # enforce a minimum hydrophobicity-distribution score
    hdistribution_threshold = Instruction(float, 0.9)     # minimum accepted distribution score

    # --- validation ----------------------------------------------------------
    check_validity = Instruction(bool, True)  # validate seed sequences (length and restrictions) before use

    iterations_elite = Instruction(int, 3)    # consecutive top generations to earn the 'elite' tag; also the reinsertion count for 'preferent'
    elite_ratio = Instruction(float, 0.01)    # fraction of the ranking counted as "top" (0.01 = top 1%)

    # --- external methods ----------------------------------------------------
    # Names of user-supplied Python modules (without .py) providing the
    # functions the Manager calls each iteration.
    constructor = Instruction(str, '')  # module with constructor_method(sequence): prepares each simulation system
    calculator = Instruction(str, '')   # module with calculator_method(sequence) and calculator_check(sequence)
    analyzer = Instruction(str, '')     # module with analyzer_method(sequence): computes and returns the fitness

    # --- evo iteration -------------------------------------------------------
    max_gen_attemps = Instruction(int, 100000)  # max attempts to generate a valid sequence before giving up
    sleep_time = Instruction(int, 1800)         # wait between simulation-status checks, in seconds
    max_check_cycle = Instruction(int, 144)     # max status-check cycles before marking pending simulations as failed
    max_generations = Instruction(int, None)    # stop after this many generations (None = no limit)
    target_fitness = Instruction(float, None)   # stop once the best fitness reaches this target (None = no target)

    ##############################

    def __init__(self, filename: str) -> None:
        """
        Build an Instructor from a YAML instruction file.

        For every option declared in `_schema`, the value is read from the YAML
        file; if it is absent (or None), the option's declared default is used.
        Assignment goes through the Instruction descriptor, so invalid values
        fall back to the default with a warning. After loading, the Generator is
        configured from the resulting attributes.

        Args:
            filename (str): Path to the YAML instruction file to read.
        """
        self.filename: str = filename
        self.yaml_data = self._load_yaml()
        ##############################
        # Populate every schema option from YAML or fall back to its default.
        for name, field in self._schema.items():
            raw = self.yaml_data.get(name, None)
            if raw is None:
                # option not present in YAML --> use the declared default
                setattr(self, name, field.default)
                continue
            try:
                setattr(self, name, raw)
            except TypeError as e:
                # invalid type/value --> warn and fall back to the default
                logging.warning(f'Default value in {name}', e, field.default)
                setattr(self, name, field.default)
        ##############################
        # Snapshot the config keys (skips filename and yaml_data) for __str__/__iter__.
        self.__config_keys = list(self.__dict__)[2:]
        self.configure_generator()
        self.cwd = os.getcwd()

    # special methods ---------------------------
    def __str__(self) -> str:
        """Return a human-readable dump of the configuration (lists shown as counts)."""
        lines = ["===== INPUT  CONFIGURATION =====\n"]
        for key in self.__config_keys:
            value = self.__dict__[key]

            # Show lists by their length rather than their full content.
            if isinstance(value, list):
                lines.append(f"{key:<26}: {len(value)}\n")
            else:
                lines.append(f"{key:<26}: {value}\n")
        lines.append("================================\n")
        return ''.join(lines)
    
    def __iter__(self):
        """Iterate over the configuration option names."""
        return iter(self.__config_keys)
    
    # read yaml
    def _load_yaml(self) -> dict:
        """Load and return the YAML file as a dict (empty dict if the file is empty)."""
        with open(self.filename, 'r', encoding='utf-8') as f:
            return yaml.load(f, Loader=yaml.FullLoader) or {}
        
    # configure and get generator 
    def configure_restrictions(self):
        """
        Build the list of active Restriction objects from the configuration.

        Each restriction is appended only when its enabling flag is set (or the
        relevant value is non-default). Restriction classes are imported lazily
        so that unused ones are never loaded.

        Returns:
            list: The Restriction instances to enforce on candidate sequences.
        """
        restrictions = []

        if self.hydrophobic_restriction:
            from restriction import HmomentRestriction
            restriction_ = HmomentRestriction(min=self.hydrophobic_min, max=self.hydrophobic_max, h_scale=self.hydrofobic_scale)
            restrictions.append(restriction_)
        
        if self.hindex_restriction:
            from restriction import HindexRestriction
            restriction_ = HindexRestriction(min=self.hindex_min, max=self.hindex_max, h_scale=self.hydrofobic_scale)
            restrictions.append(restriction_)
        
        if self.charge_restriction:
            from restriction import ChargeRestriction
            restriction_ = ChargeRestriction(min=self.charge_min, max=self.charge_max)
            restrictions.append(restriction_)
        
        if len(self.prohibited_patterns) > 0:
            from restriction import PatternRestriction
            restriction_ = PatternRestriction(patterns=self.prohibited_patterns)
            restrictions.append(restriction_)

        if self.positive_atleast > 0:
            from restriction import CompositionRestriction
            from scales import Classification
            restriction_ = CompositionRestriction(residues=Classification.positive, min=self.positive_atleast, max=self.peptide_len)
            restrictions.append(restriction_)

        if self.negative_atleast > 0:
            from restriction import CompositionRestriction
            from scales import Classification
            restriction_ = CompositionRestriction(residues=Classification.negative, min=self.negative_atleast, max=self.peptide_len)
            restrictions.append(restriction_)

        if self.hdistribution_restriction :
            from restriction import HdistributionRestriction
            restriction_ = HdistributionRestriction(min=self.hdistribution_threshold, max=None, h_scale=self.hydrofobic_scale)
            restrictions.append(restriction_)
        
        if self.excluded_sequences:
            from restriction import ForbiddenSequence
            restriction_ = ForbiddenSequence(sequences=self.excluded_sequences)
            restrictions.append(restriction_)

        return restrictions

    def _build_single_method(self, met):
        """
        Build and return a single GenMethod instance from its name.

        Method classes are imported lazily so only the requested ones are
        loaded. Returns None (with a warning) if the name is unknown.

        Args:
            met (str): Method name, one of the populate_method choices.

        Returns:
            GenMethod | None: The constructed method, or None if unrecognized.
        """
        if met == 'hybrids':
            from hybrid import Hybrid
            return Hybrid()
        if met == 'swap':
            from swap import Swap
            return Swap()
        if met == 'faces':
            from faces_mix import FacesMix
            return FacesMix(slice_angle=self.face_slice_angle)
        if met == 'group_mutation':
            from directed_mutations import GroupMutation
            return GroupMutation()
        if met == 'hydrophobic_mutation':
            from directed_mutations import HydrophobicityMutation
            return HydrophobicityMutation(h_scale=self.hydrofobic_scale)
        if met == 'pattern':
            from pattern import Pattern
            return Pattern(
                pattern=self.pattern,
                options=self.pattern_options,
                weights=self.pattern_weights,
                max_mutations=self.pattern_maxmut,
            )
        logger.warning(f"Instructor: unknown populate_method '{met}' --> ignored")
        return None

    def set_gen_methods(self):
        """
        Build the working methods, their weights, and the initial method.

        Resolves populate_method and method_weights into the objects the
        Generator needs, enforcing the combination rules:
          - A single method ignores method_weights (weight forced to 1.0).
          - Multiple methods require matching method_weights (equal if omitted).
          - 'pattern' is exclusive: it cannot be combined with other methods and
            acts as both the working and the initial (no-parent) method.
          - 'flanks' wraps an inner method (flank_inner); it becomes the sole
            method seen by the Generator and the only valid initial method.

        Returns:
            tuple: (methods, weights, initial_method), where initial_method is
            None for the regular case (Generator falls back to its random
            initial method).
        """
        # Normalize / validate weights against populate_method.
        if len(self.populate_method) == 1:
            logger.info("Instructor: Only 1 method selected. Ignoring method_weights.")
            self.method_weights = [1.]
        elif len(self.populate_method) > 1 and self.method_weights is None:
            logger.warning("Instructor: No weights received for the selected methods. "
                           "Methods will be selected with equal probability.")
            self.method_weights = [1. for _ in self.populate_method]
        elif len(self.populate_method) != len(self.method_weights):
            raise ValueError("Instructor: populate_method and method_weights "
                             "must have the same length!")

        # --- 'pattern' is exclusive --------------------------------------
        if 'pattern' in self.populate_method:
            if len(self.populate_method) > 1:
                raise ValueError(
                    "Instructor: 'pattern' cannot be combined with other "
                    f"methods, got {self.populate_method}."
                )
            pattern_method = self._build_single_method('pattern')
            # Pattern works both as the initial (no-parent) method and as the
            # single working method.
            return [pattern_method], [1.], pattern_method

        # --- 'flanks' wraps the other selected methods -------------------
        if 'flanks' in self.populate_method:
            from flanks import Flanks

            # Build the inner method that operates on the core between flanks.
            inner_methods = [self._build_single_method(self.flank_inner)]
            inner_weights = [1.]
            
            flanks_method = Flanks(
                n_flank=self.n_flank,
                c_flank=self.c_flank,
                methods=inner_methods,
                weights=inner_weights if inner_weights else None,
                extra_mutation=self.extra_mutation,
                extra_mutation_prob=self.also_mutate_probability,
            )

            # Flanks handles the extra mutation internally, so disable it here
            # to avoid applying it twice.
            self.also_mutate_probability = 0
            # Flanks is the only method the Generator sees, and also the only
            # valid no-parent initial method (it knows how to attach flanks to
            # a random core).
            return [flanks_method], [1.], flanks_method

        # --- regular case: one or more loose methods ---------------------
        methods = []
        weights = []
        for met, w in zip(self.populate_method, self.method_weights):
            built = self._build_single_method(met)
            if built is not None:
                methods.append(built)
                weights.append(w)

        if not methods:
            raise ValueError(
                f"Instructor: no valid method built from {self.populate_method}."
            )

        # No explicit initial method: Generator falls back to _RandomInitial.
        return methods, weights, None

    def configure_generator(self) -> None:
        """
        Build the Generator from the current configuration and store it.

        Assembles the active restrictions and the resolved generation methods,
        constructs a Generator with them, and assigns it to self.generator.
        """
        rest_list = self.configure_restrictions()
        methods, weights, initial = self.set_gen_methods()

        gen = Generator(
            methods=methods,
            weights=weights,
            aa_pool=self.mut_aa,
            peptide_len=self.peptide_len,
            extra_mutation=self.extra_mutation,
            extra_mutation_prob=self.also_mutate_probability,
            restrictions=rest_list,
            initial_method=initial,
        )
        self.generator = gen

if __name__ == '__main__':
    pass

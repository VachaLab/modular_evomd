# === instructor_modular.py ===
"""
This script reads an instruction file and sets attributes on the Instructor class.
The Instructor class should have default values related to the evolution process,
but not to the fitness function or specific simulation methods.

AH Tanguma
"""

import logging
from typing import Dict
import os
from utils import current_time
import yaml
from instruction_fields import Instruction
from generator import Generator

logger = logging.getLogger(__name__)


class Instructor:
    name = 'instructor'
    _schema: Dict[str, Instruction] = {}
    # Write here the valid instructions with default values and default types
    evomd_directory = Instruction(str, 'simulation_data')
    evolver_name = Instruction(str, 'evolver')
    optimize = Instruction(str, 'maximize', choices={'maximize', 'minimize'})
    sequences = Instruction(list, [], subtype=str)
    excluded_sequences = Instruction(list, [], subtype=str)  # forbidden sequences 
    
    # --- showing evolver ---
    top_list = Instruction(int, 10)  # show 10 sequences

    # --- population ---
    mut_aa = Instruction(str, 'ACDEFGHIKLMNPQRSTVWY')  # default = all natural amino acids
    peptide_len = Instruction(int, 22)  # length of peptides
    population = Instruction(int, 120)  # size of the population to be simulated
    populate_method = Instruction(
        (str, list),
        ['swap'],
        choices={'hybrids', 'group_mutation', 'hydrophobic_mutation', 'swap', 'faces', 'pattern', 'flanks'},
        choice_list=True, normalize=lambda v: [v] if isinstance(v, str) else list(v),)
    method_weights = Instruction(list, None, subtype=float)
    extra_mutation = Instruction(bool, True)  # Additional mutation based on also_mutate_probability
    also_mutate_probability = Instruction(float, 0.1, range=[0, 1])  # probability of mutating (only used if extra_mutation = true)
    include_parents = Instruction(bool, False)   # to include parents in next iteration
    include_resurrection = Instruction(bool, False)  # test again a random discarded sequence
    resurrection_probability = Instruction(float, 0.001)  # probability of resurrection instead of generate sequence
    populate_discarded = Instruction(bool, False)  # use discarded sequences to create new sequences
    avoid_reinsertion = Instruction(bool, True)  # a previously tested sequence turns into forbidden
    # --- for faces method ---
    face_slice_angle = Instruction(float, 180, range=[0, 360])  # slice angle: half of the angle on each side of hydrophobic vector
    # --- work with patterns ---
    pattern = Instruction(str, None) # '-' are positions to modify
    pattern_options = Instruction(list, None)  # available residues to change each free position '-': all mut_aa by default
    pattern_weights = Instruction(list, None)  # weights for each option: equal weights by default
    pattern_maxmut = Instruction(int, 1) # maximum number of mutations done at the same time.
    # --- work with flanks ---
    n_flank = Instruction(str, None)
    c_flank = Instruction(str, None)
    flank_inner = Instruction(str, 'swap', choices={'hybrids', 'group_mutation', 'hydrophobic_mutation', 'swap'})
        
    # choosing parents
    discard_ratio = Instruction(float, 0.7)  # A maximum of 70% of the sequences can be descarted == 30% parents --> this will be refactored as self.parent_ratio but not today
    populate_weighted = Instruction(bool, False)  # if true, better peptides have preference as parent
    weight_bias = Instruction(float, 0.3) # bias = (population - index) * weight_bias
    include_discarded = Instruction(bool, False)  # include discarded sequences in choosing parents

    # --- restrictions ---
    hydrophobic_restriction = Instruction(bool, True)  # 
    hydrophobic_min = Instruction(float, 5.5)
    hydrophobic_max = Instruction(float, +100)

    hindex_restriction = Instruction(bool, False)  # 
    hindex_min = Instruction(float, -9.0)
    hindex_max = Instruction(float, -6.5)

    charge_restriction = Instruction(bool, False)
    charge_min = Instruction(float, -100)
    charge_max = Instruction(float, +100)

    prohibited_patterns = Instruction(list, [], subtype=str)  # forbidden patterns in a sequence: ex. KKK means "three K or more together"

    positive_atleast = Instruction(int, 0)  # make valid only sequences with at least this number of positive residues
    negative_atleast = Instruction(int, 0)  # make valid only sequences with at least this number of negative residues

    hdistribution_restriction = Instruction(bool, False)
    hdistribution_threshold = Instruction(float, 0.9)
    
    # --- validation ---
    check_validity = Instruction(bool, True)  # check first sequences
    
    iterations_elite = Instruction(int, 3)  # Iterations before setting elite
    elite_ratio = Instruction(float, 0.01)  # A maximum of 1% of the sequences can be elite
    elite_bias = Instruction(float, 2.0)  # if 1 --> no bias applied in choosing method. only if populate_weighted is True

    # --- external methods ---
    constructor = Instruction(str, '')  # name of the constructor library
    calculator = Instruction(str, '')  # contains calculator and checker
    analyzer = Instruction(str, '')

    # --- evo iteration ---
    max_gen_attemps = Instruction(int, 100000)
    sleep_time = Instruction(int, 1800)  # sleep time in seconds
    max_check_cycle = Instruction(int, 144)
    max_generations = Instruction(int, None)
    convergence_criterium = Instruction(float, None)

    ##############################

    def __init__(self, filename: str) -> None:
        """
        Initialize the Instructor object with default attribute values and parse instructions from file.

        :param filename: Name of the instruction file to read.
        """
        self.filename: str = filename
        self.yaml_data = self._load_yaml()
        ##############################
        for name, field in self._schema.items():
            raw = self.yaml_data.get(name, None)
            if raw is None:
                # not in yaml --> default value
                setattr(self, name, field.default)
                continue
            try:
                setattr(self, name, raw)
            except TypeError as e:
                logging.warning(f'Default value in {name}', e, field.default)
                setattr(self, name, field.default)
        ##############################
        self.__config_keys = list(self.__dict__)[2:]
        self.configure_generator()
        self.cwd = os.getcwd()

    # special methods ---------------------------
    def __str__(self) -> str:
        lines = ["===== INPUT  CONFIGURATION =====\n"]
        for key in self.__config_keys:
            value = self.__dict__[key]

            # Formatear listas de forma especial
            if isinstance(value, list):
                lines.append(f"{key:<26}: {len(value)}\n")
            else:
                lines.append(f"{key:<26}: {value}\n")
        lines.append("================================\n")
        return ''.join(lines)
    
    def __iter__(self):
        return iter(self.__config_keys)
    
    # read yaml
    def _load_yaml(self) -> dict:
        with open(self.filename, 'r', encoding='utf-8') as f:
            return yaml.load(f, Loader=yaml.FullLoader) or {}
        
    # configure and get generator 
    def configure_restrictions(self):
        restrictions = []

        if self.hydrophobic_restriction:
            from restriction import HmomentRestriction
            restriction_ = HmomentRestriction(min=self.hydrophobic_min, max=self.hydrophobic_max)
            restrictions.append(restriction_)
        
        if self.hindex_restriction:
            from restriction import HindexRestriction
            restriction_ = HindexRestriction(min=self.hindex_min, max=self.hindex_max)
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
            restriction_ = HdistributionRestriction(min=self.hdistribution_threshold, max=None)
            restrictions.append(restriction_)
        
        if self.excluded_sequences:
            from restriction import ForbiddenSequence
            restriction_ = ForbiddenSequence(sequences=self.excluded_sequences)
            restrictions.append(restriction_)

        return restrictions

    def _build_single_method(self, met):
        """
        Builds and returns a single GenMethod instance for the method name
        `met`. Flanks is intentionally NOT handled here: it is a wrapper and
        is assembled separately in set_gen_methods() to avoid recursion.
        Returns None for unknown names.
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
            return HydrophobicityMutation()
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
        Builds the working methods and the initial (no-parent) method from
        populate_method and method_weights.

        Returns
        -------
        (methods, weights, initial_method) : tuple
            methods         : list[GenMethod] passed to Generator.methods
            weights         : list[float] aligned with methods
            initial_method  : GenMethod | None used by Generator for the
                              no-parent (first-fill) call.

        Special rules
        -------------
        - 'pattern' is exclusive: it cannot be combined with any other method.
        - 'flanks' forces the initial method to be Flanks itself, and wraps the
          remaining selected methods as its inner methods (never Flanks itself,
          so it is not recursive). The inner methods are NOT registered loose
          in the Generator; only Flanks is.
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

            # Collect every selected method except flanks itself (no recursion),
            # carrying along its weight.
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

            # reset self.also_mutate_probability to 0 after configuring flanks
            # This avoids undesired changes in flanks.
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

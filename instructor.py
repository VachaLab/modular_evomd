# === instructor.py ===
"""
Reads an instruction (YAML) file and exposes the full optimization
configuration as attributes of the Instructor class.

Instructor is the single source of configuration. It holds default values
for every option (used whenever the input file does not define them) and
builds the Generator (sequence-generation engine) and its Restriction set
from that configuration, so Evolver only needs to ask Instructor for a ready
to use Generator.

Generation methods come exclusively from generator.py / GenMethod subclasses.
Sequence-validity constraints come exclusively from restriction.py /
Restriction subclasses.

AH Tanguma
"""

import logging
from typing import Dict
import os

import yaml

from utils import current_time
from instruction_fields import Instruction, Instruction01

logger = logging.getLogger(__name__)


class Instructor:
    name = 'instructor'
    _schema: Dict[str, Instruction] = {}

    # =====================================================================
    # Valid instructions with default values and types.
    # Anything not present in the input file falls back to the default here.
    # =====================================================================

    # --- general ---------------------------------------------------------
    evomd_directory = Instruction(str, 'simulation_data')
    evolver_name = Instruction(str, 'evolver')
    optimize = Instruction(str, 'maximize', choices={'maximize', 'minimize'})

    # --- starting material ----------------------------------------------
    sequences = Instruction(list, [], subtype=str)            # seed sequences
    excluded_sequences = Instruction(list, [], subtype=str)   # never allowed

    # --- showing evolver -------------------------------------------------
    top_list = Instruction(int, 10)

    # --- population ------------------------------------------------------
    mut_aa = Instruction(str, 'ACDEFGHIKLMNPQRSTVWY')  # amino acid pool
    peptide_len = Instruction(int, 22)
    population = Instruction(int, 120)

    # Generation methods used to build the population. Both the regular
    # population and the first fill share the same set of methods. The first
    # population is always seeded from scratch (random) when there are not
    # enough parents available, this is handled by the Generator automatically.
    #
    # Available methods (mapped to GenMethod subclasses in build_generator):
    #   'hybrids'        -> Hybrid
    #   'faces'          -> FacesMix
    #   'swap'           -> Swap
    #   'group'          -> GroupMutation
    #   'hydrophobicity' -> HydrophobicityMutation
    methods = Instruction(
        list, ['hybrids', 'faces', 'swap', 'group'],
        subtype=str,
        subchoices={'hybrids', 'faces', 'swap', 'group', 'hydrophobicity'},
    )
    method_weights = Instruction(list, [1, 1, 1, 1], subtype=int)

    populate_weighted = Instruction(bool, False)   # better parents preferred
    weight_bias = Instruction(float, 0.3)          # (population - index) * weight_bias

    # extra single-point mutation applied by the Generator after the main method
    extra_mutation = Instruction(bool, True)
    extra_mutation_prob = Instruction01(0.2)

    include_discarded = Instruction(bool, False)   # discarded usable as parents
    avoid_reinsertion = Instruction(bool, True)    # tested sequence becomes forbidden

    # --- elitism / discarding -------------------------------------------
    check_validity = Instruction(bool, True)
    discard_ratio = Instruction(float, 0.7)        # up to 70% may be discarded
    iterations_elite = Instruction(int, 3)         # generations before elite is set
    elite_ratio = Instruction(float, 0.01)         # up to 1% may be elite
    elite_bias = Instruction(float, 2.0)           # bias for elite parents (weighted)

    # --- hydrophobicity scale (for reporting / Sequence) ----------------
    hydrophobic_scale = Instruction(
        str, 'eisenberg',
        choices={'eisenberg', 'eisenberg-chinos', 'kyte-doolittle',
                 'wimley-white', 'fauchere-pliska'},
    )

    # =====================================================================
    # Restrictions. Each flag enables one Restriction subclass; the *_min /
    # *_max values feed its constructor. Disabled restrictions are simply not
    # added to the Generator.
    # =====================================================================

    # Hydrophobic moment -> HmomentRestriction
    hmoment_restriction = Instruction(bool, True)
    hmoment_min = Instruction(float, 5.5)
    hmoment_max = Instruction(float, 100.0)

    # Hydrophobic index -> HindexRestriction
    hindex_restriction = Instruction(bool, False)
    hindex_min = Instruction(float, -9.0)
    hindex_max = Instruction(float, -6.5)

    # Net charge -> ChargeRestriction
    charge_restriction = Instruction(bool, False)
    charge_min = Instruction(float, -100.0)
    charge_max = Instruction(float, 100.0)

    # Composition -> CompositionRestriction (one restriction each)
    # "at least N positive residues" / "at least N negative residues"
    positive_atleast = Instruction(int, 0)
    negative_atleast = Instruction(int, 0)
    positive_residues = Instruction(str, 'RK')
    negative_residues = Instruction(str, 'DE')

    # Forbidden substrings / regex -> PatternRestriction
    prohibited_patterns = Instruction(list, [], subtype=str)

    # --- method-specific parameters -------------------------------------
    # FacesMix slice angle (degrees). 20 <= angle <= 340.
    face_slice_angle = Instruction(float, 120.0, range=[20.0, 340.0])

    # --- external method libraries (Manager) ----------------------------
    penalty = Instruction(str, '')
    apply_penalty = Instruction(str, 'always', choices={'once', 'always', 'never'})
    constructor = Instruction(str, '')
    calculator = Instruction(str, '')
    analyzer = Instruction(str, '')
    sleep_time = Instruction(int, 3600)
    max_check_cycle = Instruction(int, 48)
    #######################################################################

    def __init__(self, filename: str) -> None:
        """
        Initialize the Instructor with default values, then overwrite with
        whatever is found in the YAML instruction file.

        :param filename: path to the instruction file.
        """
        self.filename: str = filename
        self.yaml_data = self._load_yaml()
        #######################################################
        for name, field in self._schema.items():
            raw = self.yaml_data.get(name, None)
            if raw is None:
                # not in yaml --> default value
                setattr(self, name, field.default)
                continue
            try:
                setattr(self, name, raw)
            except TypeError as e:
                logger.warning(f'Default value used in {name}: {e} ({field.default})')
                setattr(self, name, field.default)
        #######################################################
        self.__config_keys = list(self.__dict__)[2:]
        self.cwd = os.getcwd()

        # normalize method weights against methods length
        self._normalize_method_weights()

    # special methods ---------------------------------------------------
    def __str__(self) -> str:
        lines = ["===== INPUT  CONFIGURATION =====\n"]
        for key in self.__config_keys:
            value = self.__dict__[key]
            if isinstance(value, list):
                lines.append(f"{key:<26}: {value if len(value) <= 6 else len(value)}\n")
            else:
                lines.append(f"{key:<26}: {value}\n")
        lines.append("================================\n")
        return ''.join(lines)

    def __iter__(self):
        return iter(self.__config_keys)

    # read yaml ---------------------------------------------------------
    def _load_yaml(self) -> dict:
        with open(self.filename, 'r', encoding='utf-8') as f:
            return yaml.load(f, Loader=yaml.FullLoader) or {}

    # configuration helpers ---------------------------------------------
    def _normalize_method_weights(self) -> None:
        """
        Ensure method_weights has the same length as methods. If they differ,
        fall back to uniform weights and warn the user.
        """
        if len(self.method_weights) != len(self.methods):
            logger.warning(
                f"Instructor: method_weights ({len(self.method_weights)}) does not "
                f"match methods ({len(self.methods)}) --> using uniform weights."
            )
            self.method_weights = [1 for _ in self.methods]

    def build_restrictions(self) -> list:
        """
        Build the list of Restriction objects from the configuration.
        Only enabled restrictions are included.
        """
        from restriction import (
            HmomentRestriction,
            HindexRestriction,
            ChargeRestriction,
            CompositionRestriction,
            PatternRestriction,
        )

        restrictions = []

        if self.hmoment_restriction:
            restrictions.append(
                HmomentRestriction(min=self.hmoment_min, max=self.hmoment_max)
            )
        if self.hindex_restriction:
            restrictions.append(
                HindexRestriction(min=self.hindex_min, max=self.hindex_max)
            )
        if self.charge_restriction:
            restrictions.append(
                ChargeRestriction(min=self.charge_min, max=self.charge_max)
            )
        if self.positive_atleast > 0:
            restrictions.append(
                CompositionRestriction(self.positive_residues, min_count=self.positive_atleast)
            )
        if self.negative_atleast > 0:
            restrictions.append(
                CompositionRestriction(self.negative_residues, min_count=self.negative_atleast)
            )
        if len(self.prohibited_patterns) > 0:
            restrictions.append(PatternRestriction(self.prohibited_patterns))

        logger.info(f"Instructor: {len(restrictions)} restriction(s) configured.")
        return restrictions

    def build_generator(self):
        """
        Build a fully configured Generator from the current configuration.

        Maps each method name to its GenMethod subclass, attaches weights,
        the amino acid pool, the target length, the extra-mutation settings
        and the restriction set. This is the single object Evolver uses to
        create new candidate sequences.
        """
        from generator import Generator
        from hybrid import Hybrid
        from faces_mix import FacesMix
        from swap import Swap
        from directed_mutations import GroupMutation, HydrophobicityMutation

        # name -> factory (callable returning a fresh GenMethod instance)
        factories = {
            'hybrids': lambda: Hybrid(),
            'faces': lambda: FacesMix(slice_angle=self.face_slice_angle),
            'swap': lambda: Swap(),
            'group': lambda: GroupMutation(),
            'hydrophobicity': lambda: HydrophobicityMutation(),
        }

        methods = []
        weights = []
        for name, weight in zip(self.methods, self.method_weights):
            factory = factories.get(name)
            if factory is None:
                logger.warning(f"Instructor: unknown method '{name}' --> skipping.")
                continue
            methods.append(factory())
            weights.append(weight)

        if not methods:
            logger.warning(
                "Instructor: no valid generation method configured --> "
                "Generator will fall back to random generation only."
            )
            methods = None
            weights = None

        generator = Generator(
            methods=methods,
            weights=weights,
            aa_pool=self.mut_aa,
            peptide_len=self.peptide_len,
            extra_mutation=self.extra_mutation,
            extra_mutation_prob=self.extra_mutation_prob,
            restrictions=self.build_restrictions(),
        )
        logger.info(f"Instructor: Generator built --> {generator}")
        return generator

    def interactive_change_method(self, generation: int):
        """
        Interactive editor for the main optimization parameters.
        Records changes to 'evo-md_changes.log' with timestamp and generation.
        """
        changes = []
        timestamp = current_time()
        log_file = os.path.join(self.cwd, 'evo-md_changes.log')

        print("=== Interactive Optimization Method Editor ===")
        print("Press 'q' at any prompt to stop.\n")

        available = ['hybrids', 'faces', 'swap', 'group', 'hydrophobicity']
        print("Available generation methods:")
        for i, method in enumerate(available):
            print(f"  {i}: {method}")
        print(f"\nCurrent methods : {self.methods}")
        print(f"Current weights : {self.method_weights}\n")

        choice = input("New methods (comma-separated names or indices) >>> ").strip()
        if choice.lower() == 'q' or not choice:
            print("No changes made.")
            return

        new_methods = []
        for token in choice.split(','):
            token = token.strip()
            if token.isdigit() and int(token) < len(available):
                new_methods.append(available[int(token)])
            elif token in available:
                new_methods.append(token)
            else:
                print(f"Ignoring invalid method '{token}'.")
        if not new_methods:
            print("No valid methods provided. No changes made.")
            return

        old_methods = self.methods
        self.methods = new_methods
        changes.append(f"methods: {old_methods} -> {self.methods}")
        print(f" -> methods set to {self.methods}")

        choice = input(
            f"New weights (comma-separated, same length) [Enter = uniform] >>> "
        ).strip()
        if choice.lower() == 'q':
            self._normalize_method_weights()
        elif choice:
            try:
                new_weights = [int(x.strip()) for x in choice.split(',')]
                old_weights = self.method_weights
                self.method_weights = new_weights
                changes.append(f"method_weights: {old_weights} -> {self.method_weights}")
            except ValueError:
                print("Invalid weights. Using uniform weights.")
                self.method_weights = [1 for _ in self.methods]
        else:
            self.method_weights = [1 for _ in self.methods]
        self._normalize_method_weights()
        print(f" -> weights set to {self.method_weights}")

        choice = input(
            f"Enable extra mutation? [True/False] [current: {self.extra_mutation}] >>> "
        ).strip()
        if choice and choice.lower() != 'q':
            old = self.extra_mutation
            self.extra_mutation = choice.lower() == 'true'
            changes.append(f"extra_mutation: {old} -> {self.extra_mutation}")
            print(f" -> extra_mutation set to {self.extra_mutation}")

            if self.extra_mutation:
                choice = input(
                    f"Set extra_mutation_prob (float 0-1) "
                    f"[current: {self.extra_mutation_prob}] >>> "
                ).strip()
                if choice and choice.lower() != 'q':
                    try:
                        old = self.extra_mutation_prob
                        self.extra_mutation_prob = float(choice)
                        changes.append(
                            f"extra_mutation_prob: {old} -> {self.extra_mutation_prob}"
                        )
                        print(f" -> extra_mutation_prob set to {self.extra_mutation_prob}")
                    except ValueError:
                        print("Invalid float. Keeping current value.")

        print("\nChanges complete.")
        if changes:
            with open(log_file, 'a') as log:
                log.write(f"--- {timestamp} [gen: {generation}] ---\n")
                for line in changes:
                    log.write(line + "\n")
                log.write("\n")
            print(f"Changes saved to log: {log_file}")


if __name__ == '__main__':
    pass

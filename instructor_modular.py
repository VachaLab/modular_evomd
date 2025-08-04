# === instructor_modular.py ===
"""
This script reads an instruction file and sets attributes on the Instructor class.
The Instructor class should have default values related to the evolution process,
but not to the fitness function or specific simulation methods.

AH Tanguma
"""

import logging
from typing import List, Union
import os
from utils import current_time
import yaml

logger = logging.getLogger(__name__)


class Instructor:
    name = 'instructor'

    def __init__(self, filename: str) -> None:
        """
        Initialize the Instructor object with default attribute values and parse instructions from file.

        :param filename: Name of the instruction file to read.
        """
        self.filename: str = filename
        with open(self.filename, 'r') as f:
            self.yaml_data = yaml.load(f, Loader=yaml.FullLoader)
        ##############################
        # Write here the valid instructions with default values and default types
        self.evomd_directory = 'simulation_data'
        self.evolver_name = 'evolver'
        self.optimize = 'maximize'
        self.sequences = []  # sequences that will be simulated 
        self.excluded_sequences = []  # forbidden sequences 
        self.prohibited_patterns = []  # forbidden patterns in a sequence: ex. KKK means "three K or more together"
        # --- showing evolver ---
        self.top_list = 10  # show 10 sequences
        # --- population ---
        self.mut_aa = 'ACDEFGHIKLMNPQRSTVWY'  # default = all natural amino acids
        self.peptide_len = 22  # length of peptides
        self.population = 120  # size of the population to be simulated
        self.populate_method = 'mixture'  # mixture, hybrids, mutations, swap, faces
        self.first_fill = 'random'  # first fill of Evolver.sequences
        self.populate_weighted = False  # if true, better peptides have preference as parent
        self.extra_mutation = True  # Additional mutation based on also_mutate_probability
        self.also_mutate_probability = 0.2  # probability of mutating (only used if extra_mutation = true)
        self.include_parents = False   # do not include parents in next iteration
        self.include_discarded = False  # include discarded sequences in choosing parents
        self.include_resurrection = False  # test again a discarded sequence
        self.avoid_reinsertion = True  # a previously tested sequence is prohibited
        self.resurrection_probability = 0.01  # probability of resurrection intead of generate sequence
        self.populate_discarded = False  # use discarded sequences to create new sequences
        self.weight_bias = 0.3 # bias = (population - index) * weight_bias
        # --- restrictions ---
        self.hydrophobic_scale = 'eisenberg'  # scale to compute hydrophobic moment: eisenberg, kyte-doolittle, wimley-white, fauchere-pliska. Hm is alway calculated.
        self.hydrophobic_restriction = True  # 
        self.hydrophobic_threshold = 5.5  #
        self.charge_restriction = False
        self.charge_min = -100.
        self.charge_max = +100.
        self.charged_extrema = False  # let N- and C- terminus be charged or not

        self.positive_atleast = 0  # make valid only sequences with at least this number of positive residues
        self.positive_preference = False
        self.positive_position = 0.
        self.positive_tolerance = 0.26

        self.negative_atleast = 0  # make valid only sequences with at least this number of negative residues
        self.negative_preference = False
        self.negative_position = -1.
        self.negative_tolerance = 0.26

        # --- for mixture method ---
        self.mixture_options = ['hybrids', 'faces', 'mutations', 'swap', ]  # mixture of population methods. Default: all the available methods but random
        self.mixture_weights = [1, 1, 1, 1]  # weights for choosing method. also_mutate_probability should be 0 if no more than 1 mutation is needed
        # --- for swap method ---
        self.maximum_swap_ratio = 0.3  # a maximum of 30 % of the sequence is swap.
        self.minimum_swap_ratio = 0.1  # a minimum of 10 % of the sequence is swap.
        self.swap_reconstruct = 'random'  # how to reconstruct the sequence? 'parent', 'random' or 'choose'
        self.swap_random_probability = 0.1  # 10% of random swap. Only works whith 'choose' 
        # --- for faces method ---
        self.face_slice_angle = 180  # slice angle: half of the angle on each side of hydrophobic vector
        self.face_reference = 'random'  # this face is taken as base, the oposite face is reconstructed: 'positive', 'negative', 'random'
        # --- ---
        self.check_validity = True  # check first sequences
        self.discard_ratio = 0.7  # A maximum of 70% of the sequences can be descarted == 30% parents --> this will be refactored as self.parent_ratio but not today
        self.iterations_elite = 3  # Iterations before setting elite
        self.elite_ratio = 0.01  # A maximum of 1% of the sequences can be elite
        self.elite_bias = 2.0  # if 1 --> no bias applied in choosing method
        # --- external methods ---
        self.penalty = ''  # name of the penalty library
        self.apply_penalty = 'always'  # once = just apply once, always = apply in each iteration
        self.constructor = ''  # name of the constructor library
        self.calculator = ''  # contains calculator and checker
        self.analyzer = ''
        self.sleep_time = 3600  # sleep time in seconds
        self.max_check_cycle = 48
        ##############################
        self.__config_keys = list(self.__dict__)[1:]
        self.__lines: List[str] = self._read_file()
        self._parse_instructions()
        self.cwd = os.getcwd()
    
    def __str__(self) -> str:
        lines = ["===== INPUT  CONFIGURATION =====\n"]
        for key in self.__config_keys:
            value = self.__dict__[key]

            # Formatear listas de forma especial
            if isinstance(value, list):
                lines.append(f"{key:<24}: {len(value)}\n")
            else:
                lines.append(f"{key:<24}: {value}\n")
        lines.append("================================\n")
        return ''.join(lines)

    def _read_file(self) -> List[str]:
        """
        Reads the instruction file and filters out empty lines and comments.

        :return: A list of valid (non-comment, non-empty) lines from the file.
        """
        if not self.filename:
            return []
        logger.info('Instructor: Reading instruction file')
        pre_lines: List[str] = []
        try:
            with open(self.filename, 'r', encoding='utf-8') as file:
                pre_lines = file.readlines()
        except IOError as e:
            logger.error(f"Instructor: Failed to read file '{self.filename}': {e}")
            raise SystemExit(1)

        lines: List[str] = []
        for line in pre_lines:
            # Remove inline comments and leading/trailing whitespace
            no_comment = line.split('#')[0]
            stripped_line = no_comment.strip()
            if not stripped_line or stripped_line.startswith('#'):
                continue
            lines.append(stripped_line)
        return lines

    def _parse_instructions(self) -> None:
        """
        Parses the cleaned instruction lines and sets corresponding attributes.

        Instructions are expected in the form of key-value pairs separated by '='.
        Values can be:
            - Single values (e.g., "start = yes")
            - Blocks enclosed in curly braces with multiple values (e.g., "penalty={0 1 2}")
            - Multi-line blocks enclosed within braces

        Values are converted to int or float where applicable.
        """
        if not self.filename:
            return 0
        instructions = {}
        i = 0

        while i < len(self.__lines):
            line = self.__lines[i]

            if '=' not in line:
                logger.warning(f'Instructor: line "{line}" does not contain any instruction --> skipping')
                i += 1
                continue

            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            if key not in self.__dict__:
                logger.warning(f'Instructor: "{key}" is not a valid instruction --> skipping')
                i += 1
                continue

            if value.startswith('{'):
                # Handle block value
                if value.endswith('}'):
                    # Single-line block
                    block_content = value[1:-1].strip()
                    tokens = block_content.split()
                    instructions[key] = [self._convert_token(token) for token in tokens]
                else:
                    # Multi-line block
                    block_tokens: List[str] = []
                    initial_tokens = value[1:].strip()
                    if initial_tokens:
                        block_tokens.extend(initial_tokens.split())

                    i += 1
                    while i < len(self.__lines):
                        current_line = self.__lines[i].strip()
                        if '}' in current_line:
                            tokens = current_line.split('}')[0].split()
                            if tokens:
                                block_tokens.extend(tokens)
                            break
                        block_tokens.extend(self.__lines[i].split())
                        i += 1
                    instructions[key] = [self._convert_token(token) for token in block_tokens]
            else:
                # Single token value
                token = value.split()[0]
                instructions[key] = self._convert_token(token)

            i += 1

        # Assign parsed values to instance attributes
        for key, value in instructions.items():
            #---------------
            if key == 'optimize':
                # verify that optimize is well defined
                if not self._verify_value(key, value, expect_type=type(self.__dict__[key]), options=['maximize', 'minimize']):
                    continue
                value = value.lower()  # normalize casing
            #---------------
            if key == 'hydrophobic_scale':
                if not self._verify_value(
                    key, value, expect_type=type(self.__dict__[key]), 
                    options=[
                        'eisenberg', 
                        'kyte-doolittle', 
                        'wimley-white', 
                        'fauchere-pliska',
                        ]
                    ):
                    continue
                value = value.lower()
            #---------------
            if key == 'populate_method' or key == 'first_fill':
                # verify that populate_method is well defined
                if not self._verify_value(
                    key, value, expect_type=type(self.__dict__[key]), 
                    options=[
                        'mixture', 
                        'hybrids',
                        'mutations',  
                        'swap',
                        'random',
                        'faces',
                        ]
                    ):
                    continue
                value = value.lower()  # normalize casing
            #---------------
            if key == 'swap_reconstruct':
                # verify that populate_method is well defined
                if not self._verify_value(
                    key, value, expect_type=type(self.__dict__[key]), 
                    options=[
                        'random', 
                        'parent',
                        'choose'
                        ]
                    ):
                    continue
                value = value.lower()  # normalize casing
            #---------------
            if key == 'face_conserved':
                # verify that populate_method is well defined
                if not self._verify_value(
                    key, value, expect_type=type(self.__dict__[key]), 
                    options=[
                        'random', 
                        'positive',
                        'negative'
                        ]
                    ):
                    continue
                value = value.lower()  # normalize casing
            #---------------
            if key == 'apply_penalty':
                # verify that optimize is well defined
                if not self._verify_value(key, value, expect_type=type(self.__dict__[key]), options=['always', 'once']):
                    continue
                value = value.lower()  # normalize casing
            #---------------
            if not self._verify_value(key, value, expect_type=type(self.__dict__[key])):
                continue
            setattr(self, key, value)
    
    def _verify_value(self, key, value, expect_type=None, options=None) -> bool:
        """
        Receives key and value and Verifies type and options
        """
        if not isinstance(value, expect_type):
            logger.warning(
                    f'Instructor: "{key}" is "{type(value).__name__}" but must be '
                    f'"{type(self.__dict__[key]).__name__}" --> using default ({self.__dict__[key]})'
                )
            return False
        if options and value.lower() not in options:
            logger.warning(f'Instructor: "{value}" is not a valid option for "optimize" --> using default ({self.__dict__[key]})')
            return False
        return True

    def _convert_token(self, token: str) -> Union[int, float, bool, str]:
        """
        Attempt to convert a token to an int, float, or bool. If conversion fails, return the original string.

        :param token: A string token from the instruction file.
        :return: Converted token as int, float, bool, or original string.
        """
        lowered = token.lower()
        if lowered == 'true':
            return True
        if lowered == 'false':
            return False
        try:
            if '.' in token:
                return float(token)
            return int(token)
        except ValueError:
            return token

    def interactive_change_method(self, generation: int):
        """
        Interactive CLI editor for key optimization parameters in Instructor.
        Records all changes to 'evo-md_changes.log' with timestamp and generation.
        
        Args:
            generation (int): Current generation number from Evolver.
        """
        changes = []
        timestamp = current_time()
        caller = os.path.basename(__file__)
        log_file = os.path.join(self.cwd, 'evo-md_changes.log')

        print("=== Interactive Optimization Method Editor ===")
        print("Choose a new method or press 'q' to exit.\n")

        # Step 1: populate_method
        populate_methods = ["mixture", "hybrids", "mutations", "swap", "faces"]
        print("Change population method:")
        for i, method in enumerate(populate_methods):
            print(f"{i}: {method}")
        choice = input(f"[current: {self.populate_method}] >>> ").strip()

        if choice.lower() == "q":
            print("No changes made.")
            return

        try:
            method_index = int(choice)
            old = self.populate_method
            self.populate_method = populate_methods[method_index]
            print(f" -> populate_method set to '{self.populate_method}'")
            changes.append(f"populate_method: {old} -> {self.populate_method}")
        except (ValueError, IndexError):
            print("Invalid input. No changes made.")
            return

        # Step 2: populate_weighted
        choice = input(f"Use weighted selection? [True/False] [current: {self.populate_weighted}] >>> ").strip()
        if choice.lower() == "q": return
        if choice:
            old = self.populate_weighted
            self.populate_weighted = choice.lower() == "true"
            changes.append(f"populate_weighted: {old} -> {self.populate_weighted}")
            print(f" -> populate_weighted set to {self.populate_weighted}")

        # Step 3: extra_mutation
        choice = input(f"Enable extra mutation? [True/False] [current: {self.extra_mutation}] >>> ").strip()
        if choice.lower() == "q": return
        if choice:
            old = self.extra_mutation
            self.extra_mutation = choice.lower() == "true"
            changes.append(f"extra_mutation: {old} -> {self.extra_mutation}")
            print(f" -> extra_mutation set to {self.extra_mutation}")

            if self.extra_mutation:
                # Step 4: also_mutate_probability
                choice = input(f"Set also_mutate_probability (float) [current: {self.also_mutate_probability}] >>> ").strip()
                if choice.lower() == "q": return
                if choice:
                    try:
                        old = self.also_mutate_probability
                        self.also_mutate_probability = float(choice)
                        changes.append(f"also_mutate_probability: {old} -> {self.also_mutate_probability}")
                        print(f" -> also_mutate_probability set to {self.also_mutate_probability}")
                    except ValueError:
                        print("Invalid float. Keeping current value.")

        # Step 5: include_resurrection
        choice = input(f"Enable resurrection of discarded sequences? [True/False] [current: {self.include_resurrection}] >>> ").strip()
        if choice.lower() == "q": return
        if choice:
            old = self.include_resurrection
            self.include_resurrection = choice.lower() == "true"
            changes.append(f"include_resurrection: {old} -> {self.include_resurrection}")
            print(f" -> include_resurrection set to {self.include_resurrection}")

            if self.include_resurrection:
                # Step 6: resurrection_probability
                choice = input(f"Set resurrection_probability (float) [current: {self.resurrection_probability}] >>> ").strip()
                if choice.lower() == "q": return
                if choice:
                    try:
                        old = self.resurrection_probability
                        self.resurrection_probability = float(choice)
                        changes.append(f"resurrection_probability: {old} -> {self.resurrection_probability}")
                        print(f" -> resurrection_probability set to {self.resurrection_probability}")
                    except ValueError:
                        print("Invalid float. Keeping current value.")

        print("\nChanges complete.")

        # LOGGING
        if changes:
            with open(log_file, 'a') as log:
                log.write(f"--- {timestamp} [gen: {generation}] ---\n")
                for line in changes:
                    log.write(line + "\n")
                log.write("\n")
            print(f"Changes saved to log: {log_file}")


if __name__ == '__main__':
    pass

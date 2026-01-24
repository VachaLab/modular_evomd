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
    prohibited_patterns = Instruction(list, [], subtype=str)  # forbidden patterns in a sequence: ex. KKK means "three K or more together"
    # --- showing evolver ---
    top_list = Instruction(int, 10)  # show 10 sequences
    # --- population ---
    mut_aa = Instruction(str, 'ACDEFGHIKLMNPQRSTVWY')  # default = all natural amino acids
    peptide_len = Instruction(int, 22)  # length of peptides
    population = Instruction(int, 120)  # size of the population to be simulated
    populate_method = Instruction(str, 'mixture', choices={'mixture', 'hybrids', 'mutations', 'swap', 'faces', 'random', 'pattern'})
    first_fill = Instruction(str, 'random', choices={'mixture', 'hybrids', 'mutations', 'swap', 'faces', 'random', 'pattern'})  # first fill of Evolver.sequences
    populate_weighted = Instruction(bool, False)  # if true, better peptides have preference as parent
    extra_mutation = Instruction(bool, True)  # Additional mutation based on also_mutate_probability
    also_mutate_probability = Instruction(float, 0.2, range=[0, 1])  # probability of mutating (only used if extra_mutation = true)
    include_parents = Instruction(bool, False)   # to include parents in next iteration
    include_discarded = Instruction(bool, False)  # include discarded sequences in choosing parents
    include_resurrection = Instruction(bool, False)  # test again a discarded sequence
    avoid_reinsertion = Instruction(bool, True)  # a previously tested sequence turns into restricted
    resurrection_probability = Instruction(float, 0.01)  # probability of resurrection instead of generate sequence
    populate_discarded = Instruction(bool, False)  # use discarded sequences to create new sequences
    weight_bias = Instruction(float, 0.3) # bias = (population - index) * weight_bias
    # --- restrictions ---
    hydrophobic_scale = Instruction(str, 'eisenberg', choices={'eisenberg', 'kyte-doolittle', 'wimley-white', 'fauchere-pliska'})  # scale to compute hydrophobic moment: eisenberg, kyte-doolittle, wimley-white, fauchere-pliska. Hm is alway calculated.
    hydrophobic_restriction = Instruction(bool, True)  # 
    hydrophobic_min = Instruction(float, 5.5)
    hydrophobic_max = Instruction(float, +100)

    hindex_restriction = Instruction(bool, False)  # 
    hindex_min = Instruction(float, -9.0)
    hindex_max = Instruction(float, -6.5)

    hdistribution_restriction = Instruction(bool, False)
    hdistribution_threshold = Instruction(float, 2.)

    charge_restriction = Instruction(bool, False)
    charge_min = Instruction(float, -100)
    charge_max = Instruction(float, +100)
    charged_extrema = Instruction(bool, False)  # let N- and C- terminus be charged or not

    positive_atleast = Instruction(int, 0)  # make valid only sequences with at least this number of positive residues
    positive_preference = Instruction(bool, False)
    positive_position = Instruction(float, 0., range=[-1, 1])
    positive_tolerance = Instruction(float, 0.26)

    negative_atleast = Instruction(int, 0)  # make valid only sequences with at least this number of negative residues
    negative_preference = Instruction(bool, False)
    negative_position = Instruction(float, -1., range=[-1, 1])
    negative_tolerance = Instruction(float, 0.26)

    # --- Activate locked positions ---
    lock_residues = False # 
    locked_positions = Instruction(list, [], subtype=int) # positions to be locked
    locked_residues = Instruction(list, [], subtype=str) # residues locked in the same order as locked_positions
    # --- work with patterns ---
    pattern = Instruction(str, '-'*peptide_len) # '-' are positions to modify
    pattern_free_positions = [k for k in pattern if k == '-']
    pattern_options = Instruction(list, [[*mut_aa]]*pattern_free_positions)  # available residues to change each free position '-': all mut_aa by default
    pattern_weights = Instruction(list, [[1]*len(mut_aa)]*pattern_free_positions)  # weights for each option: equal weights by default
    pattern_probabilities = Instruction(list, [0.3]*pattern_free_positions) # probability of change in each free position
    
    # --- for mixture method ---
    mixture_options = Instruction(list, ['hybrids', 'faces', 'mutations', 'swap', ], subtype=str, subchoices={'random', 'hybrids', 'mutations', 'swap', 'faces'})  # mixture of population methods. Default: all the available methods but random
    mixture_weights = Instruction(list, [1, 1, 1, 1], subtype=int)  # weights for choosing method. also_mutate_probability should be 0 if no more than 1 mutation is needed
    # --- for swap method ---
    minimum_swap_ratio = Instruction(float, 0.1, range=[0, 1])  # a minimum of 10 % of the sequence is swap.
    maximum_swap_ratio = Instruction(float, 0.3, range=[0, 1])  # a maximum of 30 % of the sequence is swap.
    swap_reconstruct = Instruction(str, 'random', choices={'parent', 'random', 'choose'})  # how to reconstruct the sequence? 'parent', 'random' or 'choose'
    swap_random_probability = Instruction(float, 0.1, range=[0, 1])  # 10% of random swap. Only works whith 'choose' 
    # --- for faces method ---
    face_slice_angle = Instruction(float, 180, range=[0, 360])  # slice angle: half of the angle on each side of hydrophobic vector
    face_reference = Instruction(str, 'random', choices={'positive', 'negative', 'random'})  # this face is taken as base, the oposite face is reconstructed: 'positive', 'negative', 'random'
    # --- for mutate method ---
    mutation_method = Instruction(str, 'similarity', choices={'random', 'similarity', 'hydrophobicity'})
    # --- ---
    check_validity = Instruction(bool, True)  # check first sequences
    discard_ratio = Instruction(float, 0.7)  # A maximum of 70% of the sequences can be descarted == 30% parents --> this will be refactored as self.parent_ratio but not today
    iterations_elite = Instruction(int, 3)  # Iterations before setting elite
    elite_ratio = Instruction(float, 0.01)  # A maximum of 1% of the sequences can be elite
    elite_bias = Instruction(float, 2.0)  # if 1 --> no bias applied in choosing method. only if populate_weighted is True
    # --- external methods ---
    penalty = Instruction(str, '')  # name of the penalty library
    apply_penalty = Instruction(str, 'always', choices={'once', 'always', 'never'})  # once = just apply once, always = apply in each iteration
    constructor = Instruction(str, '')  # name of the constructor library
    calculator = Instruction(str, '')  # contains calculator and checker
    analyzer = Instruction(str, '')
    sleep_time = Instruction(int, 3600)  # sleep time in seconds
    max_check_cycle = Instruction(int, 48)
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

    def interactive_change_method(self, generation: int):
        """
        Interactive editor for key optimization parameters in Instructor.
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

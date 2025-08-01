# === evolver.py ===
import logging
from sequence import Sequence
import random
import numpy as np
from manager import Manager


logger = logging.getLogger(__name__)


class Evolver:
    name = 'evolver'

    def __init__(self, instructor, recover=False) -> None:
        self.instructor = instructor
        self.manager = Manager(self)  # Manager creates directories 
        self.name = self.instructor.evolver_name
        self.sequences = []
        self.discarded_sequences = []  # to save discarded Sequences and avoid repetition
        self.parent_sequences = []
        self.to_include = []   # this list will be used to insert new sequences in the next generation
        self.excluded_sequences = []
        self.failed_sequences = []
        self.prohibited_patterns = []
        # set first sequences
        self.first_sequences()
        self.generations = 0
        self.started = False
        self.runnable = True  # used to stop optimization iteratively
        self.recover_enabled = recover
    
    # special methods ----------------------------------------
    def __len__(self):
        return len(self.sequences)
    
    def __str__(self):
        pep_len = self.instructor.peptide_len + 2
        total_sequences = len(self.sequences) + len(self.discarded_sequences) + len(self.parent_sequences)
        lines = ['===== EVOLVER CURRENT STATE =====\n']
        lines.append(f"{'Optimization':<24}: {str(self.instructor.optimize)}\n")
        is_weighted = ''
        if self.instructor.populate_weighted:
            is_weighted = 'weighted-'
        is_extra_mut = ''
        if self.instructor.extra_mutation:
            is_extra_mut = ' + mutation'
        is_resurrection = ''
        if self.instructor.include_resurrection:
            is_resurrection = ' + resurrection'
        lines.append(f"{'Population method':<24}: {is_weighted}{str(self.instructor.populate_method)}{is_extra_mut}{is_resurrection}\n")
        lines.append(f"{'Started':<24}: {str(self.started)}\n")
        lines.append(f"{'Generations':<24}: {str(self.generations)}\n")
        lines.append(f"{'Current sequences':<24}: {len(self.sequences)}\n")
        lines.append(f"{'Parent sequences':<24}: {len(self.parent_sequences)}\n")
        lines.append(f"{'Discarded sequences':<24}: {len(self.discarded_sequences)}\n")
        lines.append(f"{'Total sequences':<24}: {total_sequences}\n")
        lines.append(f"\n{f'Top {self.instructor.top_list}':<24}  {'Sequence':<{pep_len}} {'Fitness':<8} {'Hm':<8} {'Charge':<8}\n")
        all_sequences = self.parent_sequences + self.discarded_sequences + self.sequences
        i = 0
        while i < self.instructor.top_list:
            if len(all_sequences) < i+1:
                break
            seq = all_sequences[i]
            fitness = seq.get_mean_fitness()
            fitness = f"{fitness:<8.4f}" if fitness is not None else f"{'-':<8}"
            hm = f"{round(seq.hydrophobic_moment, 3)}"
            ch = f"{round(seq.charge, 1)}"
            lines.append(f"{i+1:<24}: {str(seq):<{pep_len}} {fitness:<8} {hm:<8} {ch:<8}\n")
            i += 1
        lines.append('=================================\n')
        return ''.join(lines)
    
    def __iter__(self):
        return iter(self.sequences)
    
    # files and reports ---------------------------------------------------------
    def save_pkl(self):
        from utils import save_pkl
        save_pkl(self, self.name + '.pkl')
    
    def report_sequences(self):
        """
        Write information of all sequences in a sequences_report.csv file
        """
        with open('sequences_report.csv', 'w') as fo:
            fo.write('sequence,fitness,elite,iterations,generation\n')
            all_sequences = self.parent_sequences + self.discarded_sequences + self.sequences
            for seq in all_sequences:
                fitness = seq.get_mean_fitness()
                fo.write(f'{seq.sequence},{fitness},{str(seq.is_elite)},{len(seq.fitness)},{seq.generation}\n')

    # validators ------------------------------------------------------
    def is_valid_sequence(self, seq) -> bool:
        """Check if the sequence is valid. Returns True if it is valid."""
        seq = str(seq)
        logger.debug(f'Evolver: checking validity of sequence {seq}')

        # is not valid if the sequences already exists and avoid_reinsertion is True
        if self.sequence_exists(seq) and self.instructor.avoid_reinsertion:
            logger.debug('Evolver: sequence already exists --> discarding')
            return False

        validity_value = (
            not self.has_prohibited_pattern(seq) and
            not self.is_excluded_sequence(seq) and
            not self.is_restricted(seq) and
            self.fits_length(seq)
        )

        logger.debug(f'Evolver: is sequence {seq} valid? :  {validity_value}')

        return validity_value
        
    def is_excluded_sequence(self, seq) -> bool:
        """
        True if the sequence should be excluded
        """
        seq = str(seq)
        return seq in self.excluded_sequences

    def has_prohibited_pattern(self, seq) -> bool:
        """
        True if prohibited pattern is found
        """
        seq = str(seq)
        patterns_clean = [k for k in self.prohibited_patterns if '*' not in k]
        return any(pat in seq for pat in patterns_clean)
    
    def sequence_exists(self, seq) -> bool:
        """Check if a sequence string is already present in current or discarded sequences."""
        seq = str(seq)
        all_sequences = self.sequences + self.parent_sequences + self.discarded_sequences
        sequences = [str(s) for s in all_sequences]
        return seq in sequences
    
    def is_restricted(self, seq):
        """
        Returns True if any defined restriction is true
        """
        true_values = []
        test_seq = Sequence(seq)
        positions = test_seq.get_positions()

        # Hydrophobic restrictions
        if self.instructor.hydrophobic_restriction:
            if test_seq.hydrophobic_moment < self.instructor.hydrophobic_threshold:
                true_values.append(True)
            else:
                true_values.append(False)
        # charge restrictions
        if self.instructor.charge_restriction:
            if not self.instructor.charge_min <= test_seq.charge <= self.instructor.charge_max:
                true_values.append(True)
            else:
                true_values.append(False)
        # charged extrema 
        if not self.instructor.charged_extrema:
            if test_seq.n_ter_charge + test_seq.c_ter_charge != 0:
                true_values.append(True)
            else:
                true_values.append(False)
        # positive residue position
        if self.instructor.positive_preference:
            charged_res = test_seq.get_charged_res(charge='positive')
            logger.debug(f'positive: {charged_res}')
            min_val = self.instructor.positive_position - self.instructor.positive_tolerance
            max_val = self.instructor.positive_position + self.instructor.positive_tolerance
            tester = lambda x: min_val <= x <= max_val
            true_mat = [not tester(positions[k][0]) for k in charged_res]

            check_this = [positions[k][0] for k in charged_res]
            logger.debug(f'{true_mat}   {check_this}')
            true_values.append(any(true_mat))
        
        # negative residue position
        if self.instructor.negative_preference:
            charged_res = test_seq.get_charged_res(charge='negative')
            logger.debug(f'negative: {charged_res}')
            min_val = self.instructor.negative_position - self.instructor.negative_tolerance
            max_val = self.instructor.negative_position + self.instructor.negative_tolerance
            tester = lambda x: min_val <= x <= max_val
            true_mat = [not tester(positions[k][0]) for k in charged_res]

            check_this = [positions[k][0] for k in charged_res]
            logger.debug(f'{true_mat}  {check_this}')
            true_values.append(any(true_mat))

        del test_seq
        return any(true_values)
    
    def fits_length(self, seq):
        """Checks if sequence fits the size defined by self.instructor.peptide_len"""
        if len(seq) != self.instructor.peptide_len:
            return False
        return True
    
    # decide and choose ----------------------------------------------------
    def is_top_index(self, index):
        """
        Returns true if the index is part of the elite section 
        ex. if elite is 2% of a population=128 sequences, then
        elite section covers the nex indexes [0, 1] 
        """
        num_indexes = int(self.instructor.population * self.instructor.elite_ratio)
        return index in list(range(num_indexes))

    def take_bool_decision(self, probability=0.1):
        """
        Take a decision based on the probability
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability should be between 0.0 and 1.0")
        return random.random() < probability

    def choose_sequence(
            self, weighted=False, reverse=False, exception=None, include_elite=True, 
            include_discarded=False, include_current=False,
             only_discarded=False, only_current=False,
            ):
        """Returns a Sequence object choosen randomly with or without weights."""
        # make a copy of self.sequences to avoid undesired modifications
        # population is the list from which sequence will be taken
        # as first entry: only parent sequences
        population = self.parent_sequences

        # modify population list
        # include more elements
        if include_discarded:
            population = population + self.discarded_sequences
        if include_current:
            population = population + self.sequences
        
        # select only one type of sequences
        # this section excludes the previous one
        if only_discarded:
            population = self.discarded_sequences
        if only_current:
            population = self.sequences

        # apply exceptions
        if exception:
            # if exception is not a list, tuple or set, turn it into a list
            if not isinstance(exception, (list, tuple, set)):
                exception = [exception]
            # remove exceptions
            population = [k for k in population if k not in exception]
        if not include_elite:
            # remove elite if include_elite=False
            population = [k for k in population if not k.is_elite]

        # population will no longer be modified from this point on -------------------
        if not weighted:
            # All the sequences have the same chance to be choosen
            return random.choice(population)
        
        # reverse only makes sense when weighted is True
        if reverse:
            # invert list
            population = population[::-1]

        # create weight list based on position
        weights = []
        n = len(population)
        for idx, seq in enumerate(population):
            base_weight = (n - idx) * self.instructor.weight_bias  # Mayor peso a los primeros
            if seq.is_elite:
                multiplier = self.instructor.elite_bias
                if reverse:
                    try: multiplier = 1/multiplier
                    except ZeroDivisionError: multiplier = 0.0
                base_weight *= multiplier  # Aumento o reducción del peso si es elite
            weights.append(base_weight)
        return random.choices(population, weights=weights, k=1)[0]

    def _get_populate_method(self, first=False):
        if first:
            populate_method = self.instructor.first_fill
        else:
            populate_method = self.instructor.populate_method

        # if for some reason the populate method doesnt exist...
        if not hasattr(self, f'_{populate_method}'):
            raise ValueError(f"Method '{populate_method}' does not exist.")
        
        # get populate method
        populate_method = getattr(self, f'_{populate_method}')
        return populate_method

    # creators ----------------------------------------------------
    def create_sequence(self, seq, generation=0):
        """
        General function to create Sequence objects.
        """
        h_scale = self.instructor.hydrophobic_scale
        return Sequence(seq, generation=generation, h_scale=h_scale)
    
    def random_sequence(self) -> str:
        """Generate a random peptide sequence."""
        logger.debug('---------RANDOM--------')
        son_seq = ''.join(random.choices(self.instructor.mut_aa, k=self.instructor.peptide_len))
        logger.debug(f'{son_seq} < result')
        return son_seq

    def swap_sequence(self, parent1, parent2=None, helix=True) -> str:
        """destroy a section of a sequence and reconstruct it from parent2 or randomly if parent2=None"""
        logger.debug('---------SWAP--------')
        # Be sure to have a str
        logger.debug(f'{parent1} < parent')
        
        # maximum size of the fragment
        max_size = int(self.instructor.peptide_len * self.instructor.maximum_swap_ratio)
        min_size = int(self.instructor.peptide_len * self.instructor.minimum_swap_ratio)
        
        idx1 = 0
        idx2 = 0
        fragment_len = 10000 # start always with a bigger number
        while not min_size <= fragment_len <= max_size:
            # choose two indexes
            idx1 = random.randint(0, len(parent1) - 1)
            idx2 = random.randint(0, len(parent1) - 1)
            # idx1 must be bigger than idx2
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            fragment_len = idx2 - idx1
        logger.debug('{}{}'.format(' '*idx1, '^'*fragment_len))

        # how will sequence be reconstructed?
        if parent2:
            # from parents
            new_fragment = parent2[idx1:idx2]
            if helix:
                # swap based on proximity
                logger.debug('Looking for closest residue')
                new_positions = parent2.get_positions()
                old_fragment = [k.position for k in parent1.residues[idx1:idx2]]
                new_fragment = []
                for pos in old_fragment:
                    proximity = [np.linalg.norm(k-pos) for k in new_positions]
                    min_index = np.argmin(proximity)
                    closer = parent2[min_index]
                    new_fragment.append(closer)
                    logger.debug(f'closest residue {closer} at {round(proximity[min_index], 3)}')
                new_fragment = ''.join(new_fragment)
        else:
            # randomly
            new_fragment = ''.join(random.choices(self.instructor.mut_aa, k=fragment_len))
        
        if parent2:
            logger.debug(f'{parent2} < parent2')
        else:
            logger.debug('{}{}{} < random'.format(' '*idx1, new_fragment, ' '*(self.instructor.peptide_len-idx2)))
        
        son_seq = parent1[:idx1] + new_fragment + parent1[idx2:]
        logger.debug(f'{son_seq} < result')

        return son_seq
    
    def hybridize_sequences(self, parent1, parent2) -> str:
        """Create a hybrid sequence giving priority to earlier and elite sequences."""
        logger.debug('---------HYBRIDS--------')

        # random crossover
        crossover = random.randint(1, self.instructor.peptide_len - 1)
        logger.debug(f'{parent1} < parent1')
        logger.debug('{}'.format(parent1[:crossover]))
        logger.debug(f'{parent2} < parent2')
        logger.debug('{}{}'.format(' '*crossover,parent2[crossover:]))

        son_seq = parent1[:crossover] + parent2[crossover:]
        logger.debug(f'{son_seq} < result')

        # return hybrid
        return son_seq

    def mix_faces(self, parent1, parent2) -> str:
        """
        Creates a sequence mixing the faces of two peptides. 
        'reference_face' is always the base face.
        Positive face is hydrophobic.
        """
        logger.debug('---------FACE MIX--------')
        logger.debug('{} < parent1'.format(parent1))
        logger.debug('{} < parent2'.format(parent2))

        # get faces
        pos_1, neg_1  = parent1.get_faces(phi=self.instructor.face_slice_angle)
        positions_1 = parent1.get_positions()
        positions_2 = parent2.get_positions()
        base_face = self.instructor.face_reference

        # -- choose base face ---
        if base_face == 'positive':
            logger.debug('reference is positive face')
            # positive face is taken as base
            reference_face = pos_1
            other_face = neg_1
        elif base_face == 'negative':
            logger.debug('reference is negative face')
            # negative face is taken as base
            reference_face = neg_1
            other_face = pos_1
        elif base_face == 'random':
            # base_face is selected randomly 
            if self.take_bool_decision(probability=0.5):
                logger.debug('reference is positive (random)')
                reference_face = pos_1
                other_face = neg_1
            else:
                logger.debug('reference is negative (random)')
                reference_face = neg_1 
                other_face = pos_1

        # --- create new sequence ---
        # 1. put all the aa from one_face and '-' in the other_face position
        new_sequence = [k if n in reference_face else '-' for n, k in enumerate(str(parent1))]
        logger.debug('{} < reference'.format(''.join(new_sequence)))
        logger.debug(f'{len(other_face)} positions for reconstruction')

        # 2. fill in empty spaces with equivalent positions in parent2
        # run on other_face and positions_2
        included = ['-' for k in range(self.instructor.peptide_len)]
        positions_test = [k for k in positions_2]
        sequence_2 = [k for k in parent2]
        for index in other_face:
            # distances from reference to parent2
            distances = np.array([np.linalg.norm(k-positions_1[index]) for n, k in enumerate(positions_test)])
            min_index = np.argmin(distances)
            new_sequence[index] = sequence_2[min_index]
            included[index] = sequence_2[min_index]
            positions_test = np.array([k for k in positions_test[:min_index]] + [k for k in positions_test[min_index+1:]])
            sequence_2.pop(min_index)

        logger.debug('{} < included'.format(''.join(included)))

        # are there still missing residues?
        for index, aa in enumerate(new_sequence):
            if aa == '-':
                logger.debug(f'missing residue: including {str(parent1)[index]} in position {index}')
                new_sequence[index] = parent1[index]
            else:
                continue
        
        # 3. join sequences and return
        logger.debug('{} < result'.format(''.join(new_sequence)))
        return ''.join(new_sequence)

    def mutate_sequence(self, parent) -> str:
        """
        Mutate one position of a sequence.
        If seq=None, a sequence is selected from self.sequences
        """
        logger.debug('---------MUTATION--------')
        logger.debug('{} < parent'.format(parent))

        # choose position to be mutated
        idx = random.randint(0, self.instructor.peptide_len - 1)
        logger.debug('{}^'.format(' '*idx))

        # choose new residue
        new_aa = random.choice(self.instructor.mut_aa)
        logger.debug('{}{}'.format(' '*idx, new_aa))

        # create the son sequence
        son_seq = parent[:idx] + new_aa + parent[idx + 1:]
        logger.debug('{} < result'.format(son_seq))
        
        return son_seq

    def first_sequences(self) -> None:
        logger.info('Evolver: Starting sequences')
        # First: define patterns to be excluded
        if len(self.instructor.prohibited_patterns) > 0:
            self.prohibited_patterns.extend(self.instructor.prohibited_patterns)

        # Second: Append excluded sequences to discarded
        if len(self.instructor.excluded_sequences) > 0:
            self.excluded_sequences.extend(self.instructor.excluded_sequences)

        # Third: create first sequences 
        if len(self.instructor.sequences) > 0:
            if len(self.instructor.sequences) > self.instructor.population:
                logger.warning(f'Evolver: More that {self.instructor.population} found --> taking the first {self.instructor.population}')
            for sq in self.instructor.sequences[:self.instructor.population]:
                if len(sq) != self.instructor.peptide_len:
                    # skip if len does not match
                    logger.warning(f'Evolver: Length does not match "{sq}" (expected {self.instructor.peptide_len}) --> skipping')
                    continue
                if self.instructor.check_validity:
                    if self.is_excluded_sequence(sq):
                        logger.warning(f'Evolver: Sequence "{sq}" is an excluded sequence --> skipping')
                        continue
                    elif self.has_prohibited_pattern(sq):
                        logger.warning(f'Evolver: Sequence "{sq}" has a prohibited pattern --> skipping')
                        continue
                    elif self.sequence_exists(sq):
                        logger.warning(f'Evolver: Sequence "{sq}" already exists --> skipping')
                        continue
                    elif self.is_restricted(sq):
                        logger.warning(f'Evolver: Sequence "{sq}" does not match restrictions --> skipping')
                        continue
                self.sequences.append(Sequence(sq))
    
    def include_sequences(self):
        for candidate in self.to_include:
            if len(self.sequences) == self.instructor.population:
                logger.warning('Evolver: evolver is full --> skipping external insertion')
                break
            if len(candidate) != self.instructor.peptide_len:
                    # skip if len does not match
                    logger.warning(f'Evolver: Length does not match "{candidate}" (expected {self.instructor.peptide_len}) --> not included in the next iteration')
                    continue
            if not self.is_valid_sequence(candidate):
                logger.warning(f'Evolver: Sequence {candidate} is not valid --> not included in the next iteration')
                continue
            if self.sequence_exists(candidate):
                logger.warning(f'Evolver: Sequence {candidate} was previously generated --> reinserting')
                candidate = self.take_sequence(candidate)
                candidate.check_reinsertion(iterations_preferent=self.instructor.iterations_elite)
            else:
                candidate = Sequence(candidate, generation=self.generations)
            self.sequences.append(candidate)
        # restart to_include list
        self.to_include = []

    def populate(self) -> None:
        """
        Populate self.sequences
        """
        # if it's already populated, do nothing
        if len(self.sequences) == self.instructor.population:
            logger.warning('Evolver: Already populated --> skipping')
            return
        
        # count generations once evolver is started
        if self.started and len(self.sequences) < self.instructor.population:
            self.generations += 1

        if not self.started:
            # first population
            firs_fill  = self.instructor.first_fill
            logger.info(f'Evolver: First population method is "{firs_fill}"')
            if len(self.sequences) < 2 and firs_fill not in ['random']:
                logger.warning(f'Evolver: Method "{firs_fill}" cannot be executed with less than 2 parents --> changing to "random"')
                firs_fill = 'random'
                self.instructor.first_fill = firs_fill
            populate_method = self._get_populate_method(first=True)
            self.parent_sequences = [k for k in self.sequences] # all the sequences as parent sequences in the first population
            while len(self.sequences) < self.instructor.population:
                candidate = populate_method()
                logger.info(f'New sequence: {candidate}')
                self.sequences.append(candidate)
                self.parent_sequences.append(candidate)
            self.parent_sequences = []
                
        else:
            # move failed_sequences and to_include
            try:
               len(self.to_include)
            except:
                self.to_include = []
            if len(self.to_include) > 0:
                logger.info('Evolver: External sequences found --> including sequences')
                self.include_sequences()
            # populate using populate_method
            populate_method = self._get_populate_method()
            logger.info(f'Evolver: Populating with method {self.instructor.populate_method}')
            while len(self.sequences) < self.instructor.population:
                candidate = populate_method()
                logger.info(f'New sequence: {candidate}')
                self.sequences.append(candidate)
            # discard self.parent_sequences after populate
            self.discarded_sequences = self.parent_sequences + self.discarded_sequences
            self.parent_sequences = []

    def _random(self):
        """
        Fills self.sequences completely random
        """     
        while True:
            candidate = self.random_sequence()
            if self.is_valid_sequence(candidate):
                break
        if self.sequence_exists(candidate):
            candidate = self.take_sequence(candidate)
            candidate.check_reinsertion(iterations_preferent=self.instructor.iterations_elite)
        else:
            candidate = Sequence(candidate, generation=self.generations)
        return candidate

    def _mixture(self):
        """
        Creates or takes a sequence generated by a mixture of weighted hybrids,
        mutants and random sequences.

        returns Sequence
        """
        logger.debug('Evolver: mixture method started . . . ')
        # Use probabilities from Instructor
        # choose a generation method: mutate, hybrid, faces ...
        options = self.instructor.mixture_options
        weights = self.instructor.mixture_weights
        method = random.choices(options, weights=weights, k=1)[0]
        logger.debug(f'Evolver: generating with "{method}"')
        if method == 'hybrids':
            candidate = self._hybrids()
        elif method == 'faces':
            candidate = self._faces()
        elif method == 'mutations':
            candidate = self._mutations()
        elif method == 'swap':
            candidate = self._swap()
        elif method == 'random':
            candidate = self._random()
        
        return candidate

    def _hybrids(self):
        """
        Creates linear hybrids from parents 
        """
        loop_state = True
        if self.instructor.include_resurrection and len(self.discarded_sequences) > 0:
            candidate = self._resurrection()
            loop_state = False
                
        while loop_state:
            parent1 = self.choose_sequence(
                weighted=self.instructor.populate_weighted, 
                reverse=False, include_elite=True,
                include_discarded=self.instructor.include_discarded
                )
            parent2 = self.choose_sequence(
                weighted=self.instructor.populate_weighted, 
                reverse=False, include_elite=True, exception=parent1,
                include_discarded=self.instructor.include_discarded
                )
            candidate = self.hybridize_sequences(parent1, parent2)
            if self.instructor.extra_mutation:
                # it can be also mutated (check instructor.also_mutate_probability 
                # and instructor.extra_mutation)
                also_mutate = self.take_bool_decision(probability=self.instructor.also_mutate_probability)
                if also_mutate:
                    candidate = self.mutate_sequence(candidate)
            if self.is_valid_sequence(candidate):
                break
        if self.sequence_exists(candidate):
            candidate = self.take_sequence(candidate)
            candidate.check_reinsertion(iterations_preferent=self.instructor.iterations_elite)
        else:
            candidate = Sequence(candidate, generation=self.generations)
        return candidate
    
    def _faces(self):
        """
        Creates sequences mixing faces. Check description in self.mix_faces()
        """
        loop_state = True
        if self.instructor.include_resurrection and len(self.discarded_sequences) > 0:
            candidate = self._resurrection()
            loop_state = False
                
        while loop_state:
            parent1 = self.choose_sequence(
                weighted=self.instructor.populate_weighted, 
                reverse=False, include_elite=True,
                include_discarded=self.instructor.include_discarded
                )
            parent2 = self.choose_sequence(
                weighted=self.instructor.populate_weighted, 
                reverse=False, include_elite=True, exception=parent1,
                include_discarded=self.instructor.include_discarded
                )
            candidate = self.mix_faces(parent1, parent2)
            if self.instructor.extra_mutation:
                # it can be also mutated (check instructor.also_mutate_probability 
                # and instructor.extra_mutation)
                also_mutate = self.take_bool_decision(probability=self.instructor.also_mutate_probability)
                if also_mutate:
                    candidate = self.mutate_sequence(candidate)
            if self.is_valid_sequence(candidate):
                break
        if self.sequence_exists(candidate):
            candidate = self.take_sequence(candidate)
            candidate.check_reinsertion(iterations_preferent=self.instructor.iterations_elite)
        else:
            candidate = Sequence(candidate, generation=self.generations)
        return candidate

    def _mutations(self):
        """
        Creates mutants using self.instructor.mut_aa
        """
        loop_state = True
        if self.instructor.include_resurrection and len(self.discarded_sequences) > 0:
            candidate = self._resurrection()
            loop_state = False
                
        while loop_state:
            parent = self.choose_sequence(
                weighted=self.instructor.populate_weighted, 
                reverse=False, include_elite=True,
                include_discarded=self.instructor.include_discarded
                )
            candidate = self.mutate_sequence(parent)
            if self.instructor.extra_mutation:
                # it can be also mutated (check instructor.also_mutate_probability 
                # and instructor.extra_mutation)
                also_mutate = self.take_bool_decision(probability=self.instructor.also_mutate_probability)
                if also_mutate:
                    candidate = self.mutate_sequence(candidate)
            if self.is_valid_sequence(candidate):
                break
        if self.sequence_exists(candidate):
            candidate = self.take_sequence(candidate)
            candidate.check_reinsertion(iterations_preferent=self.instructor.iterations_elite)
        else:
            candidate = Sequence(candidate, generation=self.generations)
        return candidate

    def _swap(self):
        """
        Creates sequences based on cancer method
        """
        loop_state = True
        if self.instructor.include_resurrection and len(self.discarded_sequences) > 0:
            candidate = self._resurrection()
            loop_state = False
                
        while loop_state:
            parent1 = self.choose_sequence(
                weighted=self.instructor.populate_weighted, 
                reverse=False, include_elite=True,
                include_discarded=self.instructor.include_discarded
                )
            parent2 = None
            if self.instructor.swap_reconstruct == 'parent':
                parent2 = self.choose_sequence(
                    weighted=self.instructor.populate_weighted, 
                    reverse=False, include_elite=True, exception=parent1,
                    include_discarded=self.instructor.include_discarded
                    )
            elif self.instructor.swap_reconstruct == 'choose':
                random_swap = self.take_bool_decision(probability=self.instructor.swap_random_probability)
                if not random_swap:
                    parent2 = self.choose_sequence(
                        weighted=self.instructor.populate_weighted, 
                        reverse=False, include_elite=True, exception=parent1,
                        include_discarded=self.instructor.include_discarded
                        )
                else:
                    parent2 = None
            candidate = self.swap_sequence(parent1, parent2=parent2)
            if self.instructor.extra_mutation:
                # it can be also mutated (check instructor.also_mutate_probability 
                # and instructor.extra_mutation)
                also_mutate = self.take_bool_decision(probability=self.instructor.also_mutate_probability)
                if also_mutate:
                    candidate = self.mutate_sequence(candidate)
            if self.is_valid_sequence(candidate):
                break
        if self.sequence_exists(candidate):
            candidate = self.take_sequence(candidate)
            candidate.check_reinsertion(iterations_preferent=self.instructor.iterations_elite)
        else:
            candidate = Sequence(candidate, generation=self.generations)
        return candidate
    
    def _resurrection(self):
        resurrection = self.take_bool_decision(probability=self.instructor.resurrection_probability)
        if not resurrection:
            return None
        candidate = self.choose_sequence(weighted=False, only_discarded=True)
        candidate.check_resurrection()
        return candidate

    # methods to save and restore sequences ---------------------------------
    def read_previous(self):
        """
        Reads sequences from previous simulations.
        Sequences must be in the self.instructor.evomd_directory
        Analysis method is needed.
        """
        # read all the sequences in self.instructor.evomd_directory
        import os
        import math
        self.started = True
        logger.warning('Evolver: removing sequences list to include previous calculations')
        self.sequences = []
        sequences = os.listdir(self.instructor.evomd_directory)
        for seq in sequences:
            if not self.fits_length(seq):
                # skip if length does not fit
                continue
            if not self.is_valid_sequence(seq):
                # skip is is not a valid sequence
                continue
            # create Sequence object
            new_seq = Sequence(seq)
            # set directories
            new_seq.has_directory = True
            new_seq.directory = os.path.join(self.instructor.cwd, self.instructor.evomd_directory, seq)
            new_seq.last_iter_dir = os.path.join(new_seq.directory, 'iter_1')
            # set is_waiting_analysis
            new_seq.is_waiting_analysis = True
            # append to self.sequences
            self.sequences.append(new_seq)
        # start analysis
        self.iterate(new_plan=[False, False, False, True])
        # remove fitness None and nan
        self.sequences = [k for k in self.sequences if k.get_mean_fitness() is not None and not math.isnan(k.get_mean_fitness())]
        # sort sequences
        self.sort_sequences()
        # showing sequences
        logger.info(f'Evolver: {len(self.sequences)} in sequences list')
        logger.info(f'Evolver: {len(self.parent_sequences)} in parent sequences list')
        logger.info(f'Evolver: {len(self.discarded_sequences)} in discarded sequences list')
    
    def sequence_backup(self):
        """
        Creates a json file with sequence information in Sequence.directory
        This should help in restoring optimization if evolver.pkl file is corrupted
        """
        from utils import save_json
        import os
        logger.info('Evolver: Starting sequence back up . . .')
        all_sequences = self.discarded_sequences + self.sequences + self.parent_sequences
        for seq in all_sequences:
            if not seq.has_directory:
                logger.warning(f'Evolver: sequence_backup found a sequence without directory: {str(seq)} --> creating directory')
                self.manager.create_sequence_directory(seq)
            outfile = os.path.join(seq.directory, 'sequence.json')
            save_json(seq, outfile)
        logger.info('Evolver: Sequence back up is done :)')

    # sorting and moving ----------------------------------------------------
    def sort_sequences(self) -> None:
        """
        Orders sequences according to self.instructor.optimize
        if 'maximize' --> large to small (default)
        if 'minimize' --> small to large

        It also splits the sequences into three lists:
        self.sequences will have only elite sequences
        self.discarded_sequences will contain the worst sequences
        self.parent_sequences is used to store parents before populating

        It also asigns the next values in Sequence objects:
        Sequence.is_top           true if it is in elite section
        Sequence.current_index    index in the sorted list
        """
        import math
        logger.info('Evolver: Sorting sequences')
        # work with all the sequences
        all_sequences = self.sequences + self.discarded_sequences

        # reset values
        for seq in all_sequences:
            seq.is_elite = False
            seq.is_top = False
            seq.is_discarded = True
            seq.current_index = None

        # divide in two lists to avoid order issues related to None 
        valid = [s for s in all_sequences if s.get_mean_fitness() is not None and not math.isnan(s.get_mean_fitness())]
        invalid = [s for s in all_sequences if s.get_mean_fitness() is None or math.isnan(s.get_mean_fitness())]

        reverse = True  # maximize is default
        if str(self.instructor.optimize).lower() == 'minimize':
            reverse = False

        # order only in valid elements
        valid.sort(key=lambda seq: seq.get_mean_fitness(), reverse=reverse)
        all_sequences = valid + invalid  # join the ordered list with None elements

        # split lists
        num_total = self.instructor.population
        num_elite = int(num_total * self.instructor.elite_ratio)
        num_to_keep = num_total - int(num_total * self.instructor.discard_ratio)

        # restart lists
        self.sequences = []
        self.parent_sequences = []
        self.discarded_sequences = []

        for idx, seq in enumerate(all_sequences):
            seq.current_index = idx

            # set top
            if idx < num_elite:
                seq.is_top = True

            # set elites: executed after self.instructor.iterations_elite generations
            if self.generations > self.instructor.iterations_elite:
                seq.check_elite()

            # append into lists
            if seq.is_elite:
                # elite go to self.sequences --> it's going to be simulated
                self.sequences.append(seq)
                seq.is_discarded = False
            elif len(self.parent_sequences) < (num_to_keep - len(self.sequences)):
                # non-elite sequences 
                self.parent_sequences.append(seq)
                seq.is_discarded = False
            else:
                self.discarded_sequences.append(seq)
        
        logger.debug(f'Evolver.sort_sequences(): Number of sequences: sequences {len(self.sequences)} parent {len(self.parent_sequences)} discarded {len(self.discarded_sequences)}')
    
    def set_failed(self):
        """
        move to failed_sequences is Sequence.is_failed = True
        """
        # Remove from self.sequences
        not_failed = [k for k in self.sequences if not k.is_failed]
        failed = [k for k in self.sequences if k.is_failed]
        if len(failed) > 0:
            logger.warning(f'Evolver: Failed sequences found: {len(failed)}')
        self.sequences = not_failed
        self.failed_sequences = failed

    def take_sequence(self, seq):
        """
        This should take a sequence from the lists, remove it and return it
        """
        seq = str(seq)
        existing_seq = None
        # first, look in self.discarded_sequences
        for old in self.discarded_sequences:
            if str(old) == seq:
                existing_seq = old  # get the old sequence object
                self.discarded_sequences.remove(existing_seq)  # remove from the list
                break  # stop looking for the sequence
        if not existing_seq:
            for old in self.parent_sequences:
                if str(old) == seq:
                    existing_seq = old  # get the old sequence object
                    self.discarded_sequences.remove(existing_seq)  # remove from the list
                    break  # stop looking for the sequence
        if not existing_seq:
            for old in self.sequences:
                if str(old) == seq:
                    existing_seq = old  # get the old sequence object
                    self.discarded_sequences.remove(existing_seq)  # remove from the list
                    break  # stop looking for the sequence
        if existing_seq:
            logger.info(f'Evolver: taking_sequence: sequence {seq} is taken')
            return existing_seq
        else:
            logger.error(f'Evolver: taking_sequence: trying to take {seq} but it wasnot found in any list --> stopping evolution')
            exit(2)


    def take_sequence_prev(self, seq):
        """
        Take a sequence object from self.sequences or self.discarded_sequences and remove it 
        from those lists
        """
        seq = str(seq)
        existing_seq = None
        for seq_list in [self.sequences, self.discarded_sequences]:
            for seq_old in seq_list:
                if str(seq_old) == seq:
                    existing_seq = seq_old
                    seq_list.remove(seq_old)  # remove from list
                    break
            if existing_seq:  # if it's found, stop searching
                break
        if existing_seq:
            return existing_seq

    # Iterate ----------------------------------------------------
    def is_valid_plan(self, plan):
        if not isinstance(plan, list):
            logger.warning('Evolver: new plan must be a list')
            return False
        if len(plan) != 4:
            logger.warning('Evolver: new plan must contain 4 items')
            return False
        if not all([isinstance(k, bool) for k in plan]):
            logger.warning('Evolver: new plan mut contain only bool values')
        return True

    def iterate(self, new_plan=None):
        """
        Perform one full iteration of the evolutionary cycle.
        This includes:
            1. Constructing systems for each sequence.
            2. Running simulations.
            3. Monitoring simulation completion.
            4. Analyzing results and computing fitness.
            5. Applying penalties (if defined).
            6. Marking failed simulations.
        """
        logger.info("Evolver: Starting iteration step")

        step_flags = {
            'construct': True,
            'calculate': True,
            'check': True,
            'analyze': True,
        }

        # Recovery only on first iteration of recovered session
        if self.recover_enabled:
            logger.info("Evolver: Recovery flag is set. Attempting to resume from previous interrupted session...")
            step_flags = self.manager.recover_pending_sequences(step_flags)
            self.recover_enabled = False
        
        # set a new iteration plan from arguments
        elif new_plan:
            logger.info('Evolver: new iteration plan found --> trying to set new plan')
            if self.is_valid_plan(new_plan):
                for step, new in zip(step_flags, new_plan):
                    step_flags[step] = new
        
        logger.info(f'Evolver: iteration plan: {step_flags}')

        # Step 1: Construct molecular systems (prepare input directories and files)
        if step_flags['construct']:
            logger.info("Evolver: Running constructor step")
            try:
                self.manager.run_constructors()
            except Exception as e:
                logger.error(f"Evolver: Constructor step failed with error: {e}")
                return
            # save at the end of each step
            self.save_pkl()

        # Step 2: Run simulations (submit jobs to external software)
        if step_flags['calculate']:
            logger.info("Evolver: Running calculator step")
            try:
                self.manager.run_calculators()
            except Exception as e:
                logger.error(f"Evolver: Calculator step failed with error: {e}")
                return
            self.save_pkl()

        # Step 3: Check status of running simulations
        if step_flags['check']:
            logger.info("Evolver: Checking simulation status")
            try:
                self.manager.run_checkers()
            except Exception as e:
                logger.error(f"Evolver: Checker step failed with error: {e}")
                return
            self.save_pkl()

        # Step 4: Analyze results and compute fitness
        if step_flags['analyze']:
            logger.info("Evolver: Running analyzer step")
            try:
                self.manager.run_analyzer()
            except Exception as e:
                logger.error(f"Evolver: Analyzer step failed with error: {e}")
                return
            self.save_pkl()

        # Step 5: Mark sequences with failed simulations
        self.set_failed()
        logger.info("Evolver: Iteration step completed")

    # Convergence or max generations? -----------------------------
    def check_termination(self):
        """
        Evaluate termination conditions. If met, set self.runnable = False.
        
        Criteria:
            - Manual stop via stop_evolver
            - Convergence (e.g., stable elite set or fitness variance below threshold)
            - Max generations (optional future criterion)
        """
        logger.info("Evolver: Checking termination criteria")

        # What are the convergence criteria?
        # Here we have just a maximum number of cycles
        if self.generations >= 1000:
            logger.info("Evolver: Max generations reached --> stopping")
            self.runnable = False

if __name__ == '__main__':
    pass

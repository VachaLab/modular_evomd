# === evolver.py ===
import logging
from sequence import Sequence
import random
import numpy as np
from manager import Manager
from sequence_geometry import compute_hm_scalar, compute_helix_positions


logger = logging.getLogger(__name__)


class Evolver:
    name = 'evolver'

    def __init__(
        self, instructor, recover=False,
        fast_cycle=False, verbose=True,
        ) -> None:
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
        self.fast_cycle = fast_cycle # if true, save_pkl() is avoided during iterations
        self.verbose = verbose
    
    # special methods ----------------------------------------
    def __len__(self):
        return len(self.sequences)
    
    def __str__(self):
        pep_len = self.instructor.peptide_len + 2
        total_sequences = len(self.sequences) + len(self.discarded_sequences) + len(self.parent_sequences)
        lines = ['===== EVOLVER CURRENT STATE =====\n']

        fast = ''
        if self.fast_cycle:
            fast = ' Fast'

        lines.append(f"{'Optimization':<24}: {str(self.instructor.optimize)}{fast}\n")
        
        is_weighted = ''
        if self.instructor.populate_weighted:
            is_weighted = 'weighted-'
        is_extra_mut = ''
        if self.instructor.extra_mutation:
            is_extra_mut = ' + mutation'
        
        lines.append(f"{'Population method':<24}: {is_weighted}{str(self.instructor.populate_method)}{is_extra_mut}\n")
        lines.append(f"{'Started':<24}: {str(self.started)}\n")
        lines.append(f"{'Generations':<24}: {str(self.generations)}\n")
        lines.append(f"{'Current sequences':<24}: {len(self.sequences)}\n")
        lines.append(f"{'Parent sequences':<24}: {len(self.parent_sequences)}\n")
        lines.append(f"{'Discarded sequences':<24}: {len(self.discarded_sequences)}\n")
        lines.append(f"{'Total sequences':<24}: {total_sequences}\n")
        lines.append(f"\n{f'Top {self.instructor.top_list}':<24}  {'Sequence':<{pep_len}} {'Gen':<5} {'Fitness':<8} {'Hm':<8} {'Charge':<8}\n")
        all_sequences = self.parent_sequences + self.discarded_sequences + self.sequences
        i = 0
        while i < self.instructor.top_list:
            if len(all_sequences) < i+1:
                break
            seq = all_sequences[i]
            positions = compute_helix_positions(seq)
            fitness = seq.fitness
            fitness = f"{fitness:<8.4f}" if fitness is not None else f"{'-':<8}"
            hm = f"{round(compute_hm_scalar(seq, positions), 3)}"
            gen = seq.generation
            ch = f"{round(seq.charge, 1)}"
            lines.append(f"{i+1:<24}: {str(seq):<{pep_len}} {gen:<5} {fitness:<8} {hm:<8} {ch:<8}\n")
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
            fo.write('sequence,generation,fitness\n')
            all_sequences = self.parent_sequences + self.discarded_sequences + self.sequences
            for seq in all_sequences:
                fitness = seq.fitness
                fo.write(f'{seq.sequence},{seq.generation},{fitness}\n')

    def plot_evolution(self, show_std=False, show_kids=False, name='evolution.png'):
        import math
        import matplotlib.pyplot as plt

        # get all sequences with fitness
        sequences_plot = self.parent_sequences + self.discarded_sequences
        # get all the sequences
        avail_gens = list(set([k.generation for k in sequences_plot]))

        reverse = True  # maximize is default
        if str(self.instructor.optimize).lower() == 'minimize':
            reverse = False

        # create figure
        fig, ax = plt.subplots()

        # Define lists
        ave_kids = []
        std_kids = []
        best_kid = []
        worst_kid = []
        population_fitness = []
        already_checked = []
        for g in avail_gens:
            seq_gen = [k for k in sequences_plot if k.generation == g]
            all_fit = [k.fitness for k in seq_gen]
            already_checked.extend(all_fit)
            all_fit = np.array(all_fit)
            ave_kids.append(np.mean(all_fit))
            std_kids.append(np.std(all_fit))
            best_kid.append(np.max(all_fit))
            worst_kid.append(np.min(all_fit))
            already_checked.sort(reverse=reverse)
            population_fitness.append(sum(already_checked[:self.instructor.population])/self.instructor.population)
            # include kids? Do it now!
            if show_kids:
                current_gen = [g for k in all_fit]
                ax.scatter(current_gen, all_fit, marker='o', edgecolors='lightgray', facecolors='none', alpha=0.5, s=3)
        print('-------------')
        print('Generations:', avail_gens)
        print('Population fitness:', population_fitness)
        print('-------------')
        
        # plot 
        ax.plot(avail_gens, population_fitness, color='black', linestyle='-', label='Population', linewidth=2)
        ax.plot(avail_gens, ave_kids, color='green', linestyle='-', label='Average kids', linewidth=1)
        ax.plot(avail_gens, best_kid, color='gray', linestyle='--', label='Highest fitness', linewidth=1)
        ax.plot(avail_gens, worst_kid, color='gray', linestyle='-.', label='Lowest fitness', linewidth=1)

        # set left axis
        ax.set_xlabel('Generations')
        ax.set_ylabel('Fitness')
        # ax.set_ylim(min(already_checked), max(already_checked))

        # show std?
        if show_std:
            # set right axis
            ax2 = ax.twinx()
            ax2.plot(avail_gens, std_kids, color='skyblue', linestyle=':', label='Std. dev.', linewidth=1)
            ax2.set_ylabel('Std. Dev.')
            # join legens
            lines_1, labels_1 = ax.get_legend_handles_labels()
            line_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(lines_1 + line_2, labels_1 + labels_2, loc='best')
        else:
            ax.legend(loc='best')

        # plt.xticks(np.arange(min(avail_gens), max(avail_gens)+1, 1))  # from 0 to 9, 1 by 1

        fig.tight_layout()
        plt.show()
        fig.savefig(name, dpi=300)

    # validate and find sequences ------------------------------------------------------
    def is_valid_sequence(self, seq) -> bool:
        """Check if the sequence is valid. Returns True if it is valid."""
        seq = str(seq)
        logger.debug(f'Evolver: checking validity of sequence {seq}')

        # is not valid if the sequences already exists and avoid_reinsertion is True
        if self.sequence_exists(seq) and self.instructor.avoid_reinsertion:
            logger.debug('Evolver: sequence already exists --> discarding')
            return False

        validity_value = self.instructor.generator._passes_restrictions(seq)
        
        logger.debug(f'Evolver: is sequence {seq} valid? :  {validity_value}')

        return validity_value
    
    def sequence_exists(self, seq):
        all_lists = self.sequences + self.discarded_sequences + self.parent_sequences
        if seq in all_lists:
            return True
        return False
    
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
            self, weighted: bool = False, 
            exception: (list | tuple | set| Sequence) = None,
            include_discarded: bool = False,
            only_discarded: bool = False, 
            only_current: bool = False,
            ):
        """
        Returns a Sequence object chosen randomly, optionally weighted by fitness rank.
        It assumes that sequences are already sorted by self.sort_sequences().
        """
        # define population (pool of sequences)
        # only parents is usually enough
        population = self.parent_sequences

        # include discarded sequences to choose from a bigger pool
        if include_discarded:
            population = population + self.discarded_sequences

        # choose only from discarded sequences
        if only_discarded:
            population = self.discarded_sequences
        
        # choose from self.sequences 
        if only_current:
            population = self.sequences

        # include exceptions (avoid sequences in exception iterable)
        if exception:
            if not isinstance(exception, (list, tuple, set)):
                exception = [exception]
            population = [k for k in population if k not in exception]

        # population is now fixed -------------------------------------------
        # is it weighted choice?
        if not weighted:
            return random.choice(population)

        # weight purely by rank position (better-ranked --> higher weight)
        weights = []
        n = len(population)
        for idx, seq in enumerate(population):
            weights.append((n - idx) * self.instructor.weight_bias)
        return random.choices(population, weights=weights, k=1)[0]

    # Populate -------------------------
    def first_sequences(self) -> None:
        print('Evolver: Starting sequences')
        # Create first sequences from Instructor.sequences
        if len(self.instructor.sequences) > 0:
            if len(self.instructor.sequences) > self.instructor.population:
                logger.warning(f'Evolver: More that {self.instructor.population} found --> taking the first {self.instructor.population}')
            for sq in self.instructor.sequences[:self.instructor.population]:
                if self.instructor.check_validity:
                    if len(sq) != self.instructor.peptide_len:
                        # skip if len does not match
                        logger.warning(f'Evolver: Length does not match "{sq}" (expected {self.instructor.peptide_len}) --> skipping')
                        continue
                    if not self.is_valid_sequence(sq):
                        logger.warning(f'Evolver: Sequence "{sq}" is an excluded sequence --> skipping')
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
        Populate self.sequences up to the target size.

        Children are always generated from parents that remain in
        self.parent_sequences (so the Generator has a pool to cross).

        If include_parents is True, parents are simulated too: children are
        generated only up to (population - number_of_parents), and the parents
        are then promoted into self.sequences to complete the population.
        If include_parents is False, children fill the whole population and the
        parents are moved to discarded_sequences at the end.
        """
        logger.debug("Start Evolver.populate()")

        # if it's already populated, do nothing
        if len(self.sequences) == self.instructor.population:
            logger.warning('Evolver: Already populated --> skipping')
            return

        # count generations once evolver is started
        if self.started and len(self.sequences) < self.instructor.population:
            self.generations += 1

        # --- first population: no parents given to Generator ---
        if not self.started:
            logger.debug("Evolver.start == False. Running first population.")
            print("First population.")
            while len(self.sequences) < self.instructor.population:
                candidate = self.instructor.generator.generate(
                    verbose=self.verbose, max_attempts=self.instructor.max_gen_attemps)
                if self.sequence_exists(candidate) and self.instructor.avoid_reinsertion:
                    if self.verbose:
                        print('Sequence already exists --> discarding')
                    continue
                logger.info(f'New sequence: {candidate}')
                self.sequences.append(Sequence(candidate))
            return

        # --- subsequent generations ---
        logger.debug(f"Evolver.start == True. Generation {self.generations}")

        # target number of children to generate this generation.
        # If parents are simulated too, leave room for them.
        target = self.instructor.population
        if self.instructor.include_parents:
            target = self.instructor.population - len(self.parent_sequences)

        print("Filling Population.")
        # parents stay in self.parent_sequences during generation, so the
        # Generator always has a pool to cross.
        while len(self.sequences) < target:
            parent1 = self.choose_sequence(
                weighted=self.instructor.populate_weighted,
                include_discarded=self.instructor.include_discarded,
            )
            parent2 = self.choose_sequence(
                weighted=self.instructor.populate_weighted, exception=parent1,
                include_discarded=self.instructor.include_discarded,
            )
            candidate = self.instructor.generator.generate(
                seq1=parent1, seq2=parent2,
                verbose=self.verbose, max_attempts=self.instructor.max_gen_attemps)

            already_here = self.sequence_exists(candidate)

            if already_here and self.instructor.avoid_reinsertion:
                # duplicates are forbidden --> reject
                if self.verbose:
                    print('Sequence already exists --> discarding')
                continue

            if already_here and not self.instructor.avoid_reinsertion:
                # reinsertion: recover the existing unique object, count it,
                # and move it into the current population
                existing = self.take_sequence(candidate)
                existing.check_reinsertion(
                    iterations_preferent=self.instructor.iterations_elite)
                existing.generation = self.generations
                self.sequences.append(existing)
                logger.info(f'Reinserted sequence: {candidate}')
                continue

            logger.info(f'New sequence: {candidate}')
            self.sequences.append(Sequence(candidate, generation=self.generations))

        # --- handle the parents now that generation is done ---
        if self.instructor.include_parents:
            # parents are simulated too: promote them to complete the population
            self.sequences = self.sequences + self.parent_sequences
            self.parent_sequences = []
        else:
            # parents are not simulated: discard them
            self.discarded_sequences = self.parent_sequences + self.discarded_sequences
            self.parent_sequences = []

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
            if len(seq) != self.instructor.peptide_len:
                # skip if len does not match
                logger.warning(f'Evolver: Length does not match "{seq}" (expected {self.instructor.peptide_len}) --> skipping')
                continue
            if len(seq) != self.instructor.peptide_len and self.instructor.check_validity:
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
        self.sequences = [k for k in self.sequences if k.fitness is not None and not math.isnan(k.fitness)]
        # sort sequences
        self.sort_sequences()
        # showing sequences
        logger.info(f'Evolver: {len(self.sequences)} in sequences list')
        logger.info(f'Evolver: {len(self.parent_sequences)} in parent sequences list')
        logger.info(f'Evolver: {len(self.discarded_sequences)} in discarded sequences list')
    
    def read_report(self, report_path):
        """
        Initialize the evolver from a CSV report with the format produced by
        report_sequences(): 'sequence,generation,fitness'.

        Rows whose fitness is empty or 'None' are discarded entirely.
        The starting generation is set to the highest generation among the
        loaded (fitness-bearing) sequences; populate() will then increment it
        to the next generation when the run starts.
        """
        import csv
        import math
        import os

        if not os.path.exists(report_path):
            logger.error(f'Evolver.read_report: report file not found: {report_path}')
            exit(1)

        # start from clean lists
        self.sequences = []
        self.parent_sequences = []
        self.discarded_sequences = []

        loaded = 0
        max_gen = 0

        with open(report_path, newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raw_seq = (row.get('sequence') or '').strip()
                raw_gen = (row.get('generation') or '').strip()
                raw_fit = (row.get('fitness') or '').strip()

                if not raw_seq:
                    continue

                # discard rows without a valid fitness
                if raw_fit == '' or raw_fit.lower() == 'none':
                    logger.debug(f'Evolver.read_report: discarding {raw_seq} (no fitness)')
                    continue
                try:
                    fitness = float(raw_fit)
                except ValueError:
                    logger.warning(f'Evolver.read_report: invalid fitness "{raw_fit}" for {raw_seq} --> discarding')
                    continue
                if math.isnan(fitness):
                    logger.debug(f'Evolver.read_report: discarding {raw_seq} (nan fitness)')
                    continue

                # generation
                try:
                    generation = int(raw_gen)
                except ValueError:
                    logger.warning(f'Evolver.read_report: invalid generation "{raw_gen}" for {raw_seq} --> using 0')
                    generation = 0

                # optional validity/length filters, driven by the input YAML
                if self.instructor.check_validity:
                    if len(raw_seq) != self.instructor.peptide_len:
                        logger.warning(f'Evolver.read_report: length mismatch "{raw_seq}" '
                                    f'(expected {self.instructor.peptide_len}) --> discarding')
                        continue
                    if not self.is_valid_sequence(raw_seq):
                        logger.warning(f'Evolver.read_report: invalid sequence "{raw_seq}" --> discarding')
                        continue

                # build Sequence; fitness is a read-only property, set it via fitness_list
                new_seq = Sequence(raw_seq, generation=generation)
                new_seq.fitness_list = [fitness]
                self.sequences.append(new_seq)

                loaded += 1
                if generation > max_gen:
                    max_gen = generation

        if loaded == 0:
            logger.error('Evolver.read_report: no valid sequences loaded from report')
            exit(1)

        # mark as started; leave generations at the last fitness-bearing generation
        # so that populate() increments it to the next one on --start
        self.started = True
        self.generations = max_gen

        # distribute into sequences / parent_sequences / discarded_sequences
        # and assign elite flags consistently with the rest of the code
        self.sort_sequences()

        logger.info(f'Evolver.read_report: loaded {loaded} sequences; '
                    f'current generation set to {self.generations}')
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
                logger.debug(f'Evolver: sequence_backup found a sequence without directory: {str(seq)} --> creating directory')
                self.manager.create_sequence_directory(seq)
            outfile = os.path.join(seq.directory, 'sequence.json')
            save_json(seq, outfile)
        logger.debug('Evolver: Sequence back up is done :)')
        logger.info('Evolver: Sequence back up is done')

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
        valid = [s for s in all_sequences if s.fitness is not None and not math.isnan(s.fitness)]
        invalid = [s for s in all_sequences if s.fitness is None or math.isnan(s.fitness)]

        reverse = True  # maximize is default
        if str(self.instructor.optimize).lower() == 'minimize':
            reverse = False

        # order only in valid elements
        valid.sort(key=lambda seq: seq.fitness, reverse=reverse)
        all_sequences = valid + invalid  # join the ordered list with None elements

        # split lists
        num_total = self.instructor.population
        num_to_keep = int(num_total * self.instructor.parents_ratio)

        # restart lists
        self.sequences = []
        self.parent_sequences = []
        self.discarded_sequences = []

        for idx, seq in enumerate(all_sequences):
            seq.current_index = idx

            # set top: the elite-ratio fraction at the front of the ranking
            num_elite = int(num_total * self.instructor.elite_ratio)
            seq.is_top = idx < num_elite

            # update elite tag (consecutive top generations).
            # Only meaningful once the run is past the warm-up window.
            if self.generations > self.instructor.iterations_elite:
                seq.check_elite(iterations_elite=self.instructor.iterations_elite)

            # distribution is purely by ranking
            if idx < num_to_keep:
                self.parent_sequences.append(seq)
                seq.is_discarded = False
            else:
                self.discarded_sequences.append(seq)
        
        if self.verbose:
            print("Evolver: Sequences sorted")
            print(f'Evolver.sort_sequences(): Number of sequences: sequences {len(self.sequences)} parent {len(self.parent_sequences)} discarded {len(self.discarded_sequences)}')
    

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
                    self.parent_sequences.remove(existing_seq)  # remove from the list
                    break  # stop looking for the sequence
        if not existing_seq:
            for old in self.sequences:
                if str(old) == seq:
                    existing_seq = old  # get the old sequence object
                    self.sequences.remove(existing_seq)  # remove from the list
                    break  # stop looking for the sequence
        if existing_seq:
            logger.info(f'Evolver: taking_sequence: sequence {seq} is taken')
            return existing_seq
        else:
            logger.error(f'Evolver: taking_sequence: trying to take {seq} but it wasnot found in any list --> stopping evolution')
            exit(2)

    # going back ----------------------------------------------------
    def revert_last_generation(self):
        """
        Reverts the evolver to the last fully completed generation.

        The last completed generation is defined as the highest generation
        among sequences that already have a valid fitness (not None, not nan).
        All sequences belonging to later generations are removed, and
        self.generations is set to that generation. Sequences are then
        redistributed with sort_sequences().
        """
        import math
        logger.info('Evolver: Reverting to last completed generation')

        # gather every sequence currently held (objects are unique across lists)
        all_sequences = self.sequences + self.parent_sequences + self.discarded_sequences

        # sequences with a valid (evaluated) fitness
        valid = [s for s in all_sequences
                 if s.fitness is not None and not math.isnan(s.fitness)]

        if not valid:
            logger.error('Evolver: no sequences with valid fitness found --> cannot revert')
            exit(1)

        # last completed generation = highest generation among evaluated sequences
        last_gen = max(s.generation for s in valid)
        logger.info(f'Evolver: last completed generation is {last_gen}')

        # keep only sequences from last_gen or earlier
        kept = [s for s in all_sequences
                if s.generation <= last_gen
                and s.fitness is not None and not math.isnan(s.fitness)]
        removed = len(all_sequences) - len(kept)
        logger.info(f'Evolver: removing {removed} sequences from generations after {last_gen}')

        # put everything into a single working list; sort_sequences() reads
        # self.sequences + self.discarded_sequences, so place them there
        self.sequences = kept
        self.parent_sequences = []
        self.discarded_sequences = []

        # adjust the generation counter
        self.generations = last_gen

        # redistribute into parent / discarded and refresh flags
        self.sort_sequences()

        logger.info(f'Evolver: reverted to generation {self.generations}')
        logger.info(f'Evolver: {len(self.sequences)} in sequences list')
        logger.info(f'Evolver: {len(self.parent_sequences)} in parent sequences list')
        logger.info(f'Evolver: {len(self.discarded_sequences)} in discarded sequences list')

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
            if not self.fast_cycle:
                self.save_pkl()

        # Step 2: Run simulations (submit jobs to external software)
        if step_flags['calculate']:
            logger.info("Evolver: Running calculator step")
            try:
                self.manager.run_calculators()
            except Exception as e:
                logger.error(f"Evolver: Calculator step failed with error: {e}")
                return
            if not self.fast_cycle:
                self.save_pkl()

        # Step 3: Check status of running simulations
        if step_flags['check']:
            logger.info("Evolver: Checking simulation status")
            try:
                self.manager.run_checkers()
            except Exception as e:
                logger.error(f"Evolver: Checker step failed with error: {e}")
                return
            if not self.fast_cycle:
                self.save_pkl()

        # Step 4: Analyze results and compute fitness
        if step_flags['analyze']:
            logger.info("Evolver: Running analyzer step")
            try:
                self.manager.run_analyzer()
            except Exception as e:
                logger.error(f"Evolver: Analyzer step failed with error: {e}")
                return
            if not self.fast_cycle:
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
        if self.instructor.max_generations is not None:
            if self.generations >= self.instructor.max_generations:
                logger.info("Evolver: Max generations reached --> stopping")
                self.runnable = False
        
        if self.instructor.target_fitness is not None:
            minimize = str(self.instructor.optimize).lower() == 'minimize'
            all_seqs = self.sequences + self.parent_sequences + self.discarded_sequences
            fitnesses = [s.fitness for s in all_seqs if s.fitness is not None]

            if fitnesses:
                best = min(fitnesses) if minimize else max(fitnesses)
                target = self.instructor.target_fitness
                reached = (best <= target) if minimize else (best >= target)
                if reached:
                    logger.info("Evolver: Target fitness reached --> stopping")
                    self.runnable = False


if __name__ == '__main__':
    pass

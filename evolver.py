# === evolver.py ===
import logging
from sequence import Sequence
import random
import numpy as np
from manager import Manager


logger = logging.getLogger(__name__)


class Evolver:
    name = 'evolver'

    def __init__(self, instructor, recover=False, fast_cycle=False, verbose=True) -> None:
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
        lines.append(f"\n{f'Top {self.instructor.top_list}':<24}  {'Sequence':<{pep_len}} {'Fitness':<8} {'Hm':<8} {'Hi':<8} {'Charge':<8}\n")
        all_sequences = self.parent_sequences + self.discarded_sequences + self.sequences
        i = 0
        while i < self.instructor.top_list:
            if len(all_sequences) < i+1:
                break
            seq = all_sequences[i]
            fitness = seq.get_mean_fitness()
            fitness = f"{fitness:<8.4f}" if fitness is not None else f"{'-':<8}"
            hm = f"{round(seq.hydrophobic_moment, 3)}"
            hi = f"{seq.hydrophobic_index}"
            ch = f"{round(seq.charge, 1)}"
            lines.append(f"{i+1:<24}: {str(seq):<{pep_len}} {fitness:<8} {hm:<8} {hi:<8} {ch:<8}\n")
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
            fo.write('sequence,generation,Hm,Hi,fitness\n')
            all_sequences = self.parent_sequences + self.discarded_sequences + self.sequences
            for seq in all_sequences:
                fitness = seq.get_mean_fitness()
                fo.write(f'{seq.sequence},{seq.generation},{seq.hydrophobic_moment},{seq.hydrophobic_index},{fitness}\n')

    def plot_evolution(self, show_std=False, show_kids=False):
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
            all_fit = [k.get_mean_fitness() for k in seq_gen]
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
        ax.plot(avail_gens, ave_kids, color='green', linestyle='-', label='Av. fitness', linewidth=1)
        ax.plot(avail_gens, best_kid, color='gray', linestyle='--', label='Best fitness', linewidth=1)
        ax.plot(avail_gens, worst_kid, color='gray', linestyle='-.', label='Worst fitness', linewidth=1)

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
        fig.savefig('evolution.png', dpi=300)

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
        Populate self.sequences
        """
        logger.debug("Start Evolver.populate()")
        # if it's already populated, do nothing
        if len(self.sequences) == self.instructor.population:
            logger.warning('Evolver: Already populated --> skipping')
            return
        
        # count generations once evolver is started
        if self.started and len(self.sequences) < self.instructor.population:
            self.generations += 1

        # First population if Evolver is not started
        # No parents given to Generator
        if not self.started:
            logger.debug("Evolver.start == False. Running first population.")
            # first population
            print("First population.")
            while len(self.sequences) < self.instructor.population:
                candidate = self.instructor.generator.generate(verbose=self.verbose, max_attempts=self.instructor.max_gen_attemps)
                # is not valid if the sequences already exists and avoid_reinsertion is True
                if self.sequence_exists(candidate) and self.instructor.avoid_reinsertion:
                    print('Sequence already exists --> discarding')
                    continue
                logger.info(f'New sequence: {candidate}')
                self.sequences.append(Sequence(candidate))
                
        else:
            logger.debug(f"Evolver.start == True. Generation {self.generations}")
            # populate using populate_method
            print("Filling Population.")
            while len(self.sequences) < self.instructor.population:
                # Choose parents: always 2?
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
                candidate = self.instructor.generator.generate(seq1=parent1, seq2=parent2, verbose=self.verbose, max_attempts=self.instructor.max_gen_attemps)
                # is not valid if the sequences already exists and avoid_reinsertion is True
                if self.sequence_exists(candidate) and self.instructor.avoid_reinsertion:
                    print('Sequence already exists --> discarding')
                    continue
                logger.info(f'New sequence: {candidate}')
                self.sequences.append(Sequence(candidate, generation=self.generations))
            # discard self.parent_sequences after populate
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
            if not self.fits_length(seq):
                # skip if length does not fit
                continue
            if not self.is_valid_sequence(seq) and self.instructor.check_validity:
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
        if self.generations >= 1000:
            logger.info("Evolver: Max generations reached --> stopping")
            self.runnable = False

if __name__ == '__main__':
    pass

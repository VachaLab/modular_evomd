# === evolver.py ===
"""
Core of the evolutionary optimizer: the :class:`Evolver` class.

An Evolver holds the population of peptide :class:`Sequence` objects and
drives the genetic-algorithm cycle: populate -> simulate/evaluate ->
sort -> repopulate, generation after generation. It delegates all the
filesystem and simulation work to a :class:`Manager`, and reads its
configuration from an :class:`Instructor` (built from the input YAML).

Population lists
----------------
The population is split across three lists, rearranged every generation
by :meth:`Evolver.sort_sequences`:

    * ``self.sequences``          : elite sequences (re-simulated each generation)
    * ``self.parent_sequences``   : sequences kept as parents for breeding
    * ``self.discarded_sequences``: the rest (worst-ranked / already evaluated)

State is persisted to ``<name>.pkl`` (default ``evolver.pkl``) via
:meth:`Evolver.save_pkl`, so a run can be resumed after interruption.
"""
import logging
from sequence import Sequence
import random
import numpy as np
from manager import Manager
from sequence_geometry import compute_hm_scalar, compute_helix_positions


logger = logging.getLogger(__name__)


class Evolver:
    """Genetic-algorithm optimizer over a population of peptide Sequences.

    Parameters
    ----------
    instructor : Instructor
        Holds all configuration (population size, ratios, optimization
        direction, stop criteria, the sequence generator, etc.).
    recover : bool
        If True, the next iteration tries to resume pending sequences from
        an interrupted session (see :meth:`iterate`).
    fast_cycle : bool
        If True, intermediate ``save_pkl`` calls are skipped; the pickle is
        only written at the end of the run.
    verbose : bool
        Forwarded to the sequence generator for verbose logging.

    Key attributes
    --------------
    sequences, parent_sequences, discarded_sequences : list[Sequence]
        The three population lists (see module docstring).
    generations : int
        Current generation counter.
    started : bool
        False until the first population has been built; controls whether
        ``populate`` breeds from parents or generates from scratch.
    runnable : bool
        Main-loop flag; set to False to stop the optimization.
    """

    name = 'evolver'

    def __init__(self, instructor, recover: bool = False,
                 fast_cycle: bool = False, verbose: bool = True) -> None:
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
    def __len__(self) -> int:
        """Number of sequences currently in the active ``sequences`` list."""
        return len(self.sequences)
    
    def __str__(self) -> str:
        """Human-readable summary: config, counts and the top-N sequences."""
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
        """Iterate over the active ``sequences`` list."""
        return iter(self.sequences)
    
    # files and reports ---------------------------------------------------------
    def save_pkl(self) -> None:
        """Persist the whole Evolver state to ``<name>.pkl``."""
        from utils import save_pkl
        save_pkl(self, self.name + '.pkl')
    
    def report_sequences(self) -> None:
        """Write every sequence to ``sequences_report.csv``.

        Columns are ``sequence,generation,fitness`` (one row per sequence
        across all three population lists). This is the same format that
        :meth:`read_report` consumes.
        """
        with open('sequences_report.csv', 'w') as fo:
            fo.write('sequence,generation,fitness\n')
            all_sequences = self.parent_sequences + self.discarded_sequences + self.sequences
            for seq in all_sequences:
                fitness = seq.fitness
                fo.write(f'{seq.sequence},{seq.generation},{fitness}\n')

    def plot_evolution(self, show_std: bool = False, show_kids: bool = False) -> None:
        """Plot fitness over generations and save it to ``evolution.png``.

        Draws the running population fitness, the per-generation average,
        best and worst fitness. Optionally overlays the standard deviation
        on a second axis and/or scatters every individual ("kid").

        Parameters
        ----------
        show_std : bool
            If True, add the per-generation standard deviation on a twin axis.
        show_kids : bool
            If True, scatter every sequence's fitness at its generation.
        """
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
        """Return True if ``seq`` is an acceptable sequence.

        A sequence is rejected if it already exists and reinsertion is
        disabled, or if it fails the generator's composition/restriction
        checks.

        Parameters
        ----------
        seq : str | Sequence
            Sequence to validate.
        """
        seq = str(seq)
        logger.debug(f'Evolver: checking validity of sequence {seq}')

        # is not valid if the sequences already exists and avoid_reinsertion is True
        if self.sequence_exists(seq) and self.instructor.avoid_reinsertion:
            logger.debug('Evolver: sequence already exists --> discarding')
            return False

        validity_value = self.instructor.generator._passes_restrictions(seq)
        
        logger.debug(f'Evolver: is sequence {seq} valid? :  {validity_value}')

        return validity_value
    
    def sequence_exists(self, seq) -> bool:
        """Return True if ``seq`` is already present in any population list."""
        all_lists = self.sequences + self.discarded_sequences + self.parent_sequences
        if seq in all_lists:
            return True
        return False
    
    # decide and choose ----------------------------------------------------
    def is_top_index(self, index: int) -> bool:
        """Return True if ``index`` falls inside the elite section.

        Example: with a population of 128 and a 2% elite ratio, the elite
        section covers indices [0, 1].
        """
        num_indexes = int(self.instructor.population * self.instructor.elite_ratio)
        return index in list(range(num_indexes))

    def take_bool_decision(self, probability: float = 0.1) -> bool:
        """Return True with the given ``probability`` (a coin flip).

        Parameters
        ----------
        probability : float
            Probability of returning True, in [0.0, 1.0].
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability should be between 0.0 and 1.0")
        return random.random() < probability

    def choose_sequence(
            self, weighted=False, reverse=False, exception=None, include_elite=True, 
            include_discarded=False, include_current=False,
             only_discarded=False, only_current=False,
            ):
        """Pick one Sequence at random from a configurable candidate pool.

        The candidate pool starts as ``parent_sequences`` and can be widened
        (``include_*``), narrowed to a single list (``only_*``), or filtered
        (``exception``, ``include_elite``). When ``weighted`` is True the
        choice is biased toward better-ranked (lower-index) sequences, with
        an extra multiplier for elites.

        Parameters
        ----------
        weighted : bool
            If True, weight the choice by rank instead of choosing uniformly.
        reverse : bool
            Only meaningful when ``weighted``; flips the bias toward the
            worst-ranked sequences (and inverts the elite multiplier).
        exception : Sequence | iterable[Sequence] | None
            Sequence(s) to exclude from the pool (e.g. an already-chosen parent).
        include_elite : bool
            If False, drop elite sequences from the pool.
        include_discarded, include_current : bool
            Widen the pool with the discarded / current sequences.
        only_discarded, only_current : bool
            Restrict the pool to that single list (overrides the include_* widening).

        Returns
        -------
        Sequence
            The chosen sequence.
        """
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
        """Seed the initial population from ``instructor.sequences`` (if any).

        Called once at construction. Sequences supplied in the YAML are
        added (up to ``population``), optionally skipping those with the
        wrong length or that fail validation when ``check_validity`` is on.
        """
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
    
    def include_sequences(self) -> None:
        """Inject the sequences queued in ``self.to_include`` into the population.

        Used by the ``--insert-sequence`` action. Each candidate is checked
        for length and validity; if it already exists it is taken back from
        its list and flagged for reinsertion, otherwise a new Sequence is
        created. The queue is cleared at the end.
        """
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
        """Fill ``self.sequences`` up to the target population size.

        Two regimes:

        * Not started yet (``self.started`` is False): generate sequences
          from scratch (no parents) until the population is full.
        * Started: increment the generation counter and breed offspring by
          choosing parents with :meth:`choose_sequence` and combining them
          with the configured generator. After breeding, the previous
          parents are moved into ``discarded_sequences``.

        Does nothing if the population is already full.
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
    def read_previous(self) -> None:
        """Rebuild the population from simulations already on disk.

        Scans ``instructor.evomd_directory`` for sequence folders, wraps each
        one in a Sequence pointing at its directory, flags them for analysis,
        runs only the analyze step (``iterate`` with an analyze-only plan),
        drops sequences with no/NaN fitness and finally sorts the population.

        Used by the ``--populate-previous`` action.
        """
        # read all the sequences in self.instructor.evomd_directory
        import os
        import math
        self.started = True
        logger.warning('Evolver: removing sequences list to include previous calculations')
        self.sequences = []
        sequences = os.listdir(self.instructor.evomd_directory)
        for seq in sequences:
            # NOTE: `seq` is the folder name (a str). The two checks below
            # guard length/validity; the second references `sq` (see FIXME).
            if not self.fits_length(seq):
                # skip if length does not fit
                continue
            if len(sq) != self.instructor.peptide_len and self.instructor.check_validity:
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
    
    def read_report(self, report_path: str) -> None:
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
    
    def sequence_backup(self) -> None:
        """Write a ``sequence.json`` into each sequence's directory.

        A redundant, per-sequence backup that helps recover the run if the
        main ``evolver.pkl`` gets corrupted. Creates the directory first if
        a sequence does not have one yet.
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
        """Rank the population by fitness and split it into the three lists.

        Ordering follows ``instructor.optimize`` ('maximize' -> high-to-low,
        the default; 'minimize' -> low-to-high). Sequences without a valid
        fitness (None/NaN) are pushed to the end.

        After ranking, sequences are distributed as follows:

            * elite sequences        -> ``self.sequences`` (re-simulated)
            * next best (parents)    -> ``self.parent_sequences``
            * the rest               -> ``self.discarded_sequences``

        Per-sequence flags are refreshed in the process:
            ``is_top``        : True if inside the elite section (by rank).
            ``is_elite``      : set via ``check_elite`` once past
                                ``iterations_elite`` generations.
            ``is_discarded``  : True unless kept as elite or parent.
            ``current_index`` : position in the sorted list.
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
        num_elite = int(num_total * self.instructor.elite_ratio)
        num_to_keep = int(num_total * self.instructor.parents_ratio)

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
    
    def set_failed(self) -> None:
        """Move sequences flagged ``is_failed`` out of the active list.

        Failed sequences are removed from ``self.sequences`` and collected
        into ``self.failed_sequences``.
        """
        # Remove from self.sequences
        not_failed = [k for k in self.sequences if not k.is_failed]
        failed = [k for k in self.sequences if k.is_failed]
        if len(failed) > 0:
            logger.warning(f'Evolver: Failed sequences found: {len(failed)}')
        self.sequences = not_failed
        self.failed_sequences = failed

    def take_sequence(self, seq):
        """Find ``seq`` in any population list, remove it and return it.

        Searches ``discarded_sequences``, then ``parent_sequences``, then
        ``sequences``. If found, the existing Sequence object is removed from
        its list and returned (so it can be reinserted). If not found, the
        run is aborted (``exit(2)``).

        Parameters
        ----------
        seq : str | Sequence
            The sequence to take.

        Returns
        -------
        Sequence
            The removed Sequence object.
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
        """Older variant of :meth:`take_sequence` (kept for compatibility).

        Searches only ``sequences`` and ``discarded_sequences``, removes the
        match and returns it; returns None if not found.
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
    def is_valid_plan(self, plan) -> bool:
        """Validate an iteration plan: a list of exactly 4 booleans.

        The plan toggles the four iteration steps
        (construct, calculate, check, analyze). Returns False if the shape
        is wrong.
        """
        if not isinstance(plan, list):
            logger.warning('Evolver: new plan must be a list')
            return False
        if len(plan) != 4:
            logger.warning('Evolver: new plan must contain 4 items')
            return False
        if not all([isinstance(k, bool) for k in plan]):
            logger.warning('Evolver: new plan mut contain only bool values')
        return True

    def iterate(self, new_plan=None) -> None:
        """Run one full pass of the evolutionary cycle.

        Steps (each can be toggled via an iteration plan):
            1. construct : build the system / input files for each sequence.
            2. calculate : submit / run the simulations.
            3. check     : monitor simulation status.
            4. analyze   : read results and compute fitness.
        After the plan runs, failed sequences are moved aside
        (:meth:`set_failed`).

        Behaviour notes:
            * If ``recover_enabled`` is set, the first call resumes pending
              sequences from an interrupted session and clears the flag.
            * Otherwise, if ``new_plan`` is given (a 4-bool list validated by
              :meth:`is_valid_plan`), it overrides which steps run; e.g.
              ``[False, False, False, True]`` runs only the analyze step.
            * Each completed step saves the pickle unless ``fast_cycle``.

        Parameters
        ----------
        new_plan : list[bool] | None
            Optional [construct, calculate, check, analyze] toggle list.
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
    def check_termination(self) -> None:
        """Stop the run if a termination criterion is met (sets ``runnable``).

        Two criteria are evaluated (both read from the input YAML; each is
        skipped when left as None):

            * ``max_generations`` : stop once ``generations`` reaches it.
            * ``target_fitness``  : stop once the target fitness is reached.

        A manual stop is handled separately by the ``--stop-evolver`` action,
        which sets ``runnable = False`` directly.

        Note
        ----
        The ``target_fitness`` check inspects ``discarded_sequences[0]`` and
        assumes 'maximize'; it can raise IndexError if that list is empty and
        does not account for 'minimize'. Left as-is here (documentation pass).
        """
        logger.info("Evolver: Checking termination criteria")

        # What are the convergence criteria?
        # Here we have just a maximum number of cycles
        if self.instructor.max_generations is not None:
            if self.generations >= self.instructor.max_generations:
                logger.info("Evolver: Max generations reached --> stopping")
                self.runnable = False
        if self.instructor.target_fitness is not None:
            if self.discarded_sequences[0].fitness >= self.instructor.target_fitness:
                logger.info("Evolver: Target fitness reached --> stopping")
                self.runnable = False


if __name__ == '__main__':
    pass

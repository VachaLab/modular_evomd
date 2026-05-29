# === evolver.py ===
import logging
import math
import random

import numpy as np

from sequence import Sequence
from manager import Manager
from sequence_geometry import compute_helix_positions, compute_hm_scalar


logger = logging.getLogger(__name__)


def _hm_of(seq) -> float:
    """
    Compute the hydrophobic moment of a Sequence using sequence_geometry.
    Used only for reporting/display, since Sequence no longer stores it.
    """
    try:
        positions = compute_helix_positions(seq, translate=False)
        return round(float(compute_hm_scalar(seq, positions)), 3)
    except Exception as e:  # never let reporting break the run
        logger.debug(f"Evolver: could not compute Hm for {seq}: {e}")
        return float('nan')


class Evolver:
    name = 'evolver'

    def __init__(self, instructor, recover=False) -> None:
        self.instructor = instructor
        self.manager = Manager(self)  # Manager creates directories
        self.name = self.instructor.evolver_name

        # sequence lists
        self.sequences = []
        self.discarded_sequences = []   # tested sequences kept to avoid repetition
        self.parent_sequences = []      # parents held before populating
        self.to_include = []            # external sequences for the next generation
        self.excluded_sequences = []    # never allowed
        self.failed_sequences = []
        self.prohibited_patterns = []

        # generation engine (built from the configuration)
        self.generator = self.instructor.build_generator()

        # set first sequences
        self.first_sequences()

        self.generations = 0
        self.started = False
        self.runnable = True
        self.recover_enabled = recover

    # special methods ----------------------------------------
    def __len__(self):
        return len(self.sequences)

    def __str__(self):
        pep_len = self.instructor.peptide_len + 2
        total_sequences = (
            len(self.sequences) + len(self.discarded_sequences) + len(self.parent_sequences)
        )
        lines = ['===== EVOLVER CURRENT STATE =====\n']
        lines.append(f"{'Optimization':<24}: {str(self.instructor.optimize)}\n")
        is_extra_mut = ' + mutation' if self.instructor.extra_mutation else ''
        is_weighted = 'weighted ' if self.instructor.populate_weighted else ''
        lines.append(
            f"{'Methods':<24}: {is_weighted}{self.instructor.methods}{is_extra_mut}\n"
        )
        lines.append(f"{'Started':<24}: {str(self.started)}\n")
        lines.append(f"{'Generations':<24}: {str(self.generations)}\n")
        lines.append(f"{'Current sequences':<24}: {len(self.sequences)}\n")
        lines.append(f"{'Parent sequences':<24}: {len(self.parent_sequences)}\n")
        lines.append(f"{'Discarded sequences':<24}: {len(self.discarded_sequences)}\n")
        lines.append(f"{'Total sequences':<24}: {total_sequences}\n")
        lines.append(
            f"\n{f'Top {self.instructor.top_list}':<24}  "
            f"{'Sequence':<{pep_len}} {'Fitness':<8} {'Hm':<8} {'Hi':<8} {'Charge':<8}\n"
        )
        all_sequences = self.parent_sequences + self.discarded_sequences + self.sequences
        i = 0
        while i < self.instructor.top_list:
            if len(all_sequences) < i + 1:
                break
            seq = all_sequences[i]
            fitness = seq.fitness
            fitness = f"{fitness:<8.4f}" if fitness is not None else f"{'-':<8}"
            hm = f"{_hm_of(seq)}"
            hi = f"{seq.hydrophobic_index}"
            ch = f"{round(seq.charge, 1)}"
            lines.append(
                f"{i + 1:<24}: {str(seq):<{pep_len}} {fitness:<8} {hm:<8} {hi:<8} {ch:<8}\n"
            )
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
        """Write information of all sequences in a sequences_report.csv file."""
        with open('sequences_report.csv', 'w') as fo:
            fo.write('sequence,generation,Hm,Hi,fitness\n')
            all_sequences = self.parent_sequences + self.discarded_sequences + self.sequences
            for seq in all_sequences:
                fo.write(
                    f'{seq.sequence},{seq.generation},{_hm_of(seq)},'
                    f'{seq.hydrophobic_index},{seq.fitness}\n'
                )

    def plot_evolution(self, show_std=False, show_kids=False):
        import matplotlib.pyplot as plt

        sequences_plot = self.parent_sequences + self.discarded_sequences
        avail_gens = sorted(set(k.generation for k in sequences_plot))

        reverse = True  # maximize is default
        if str(self.instructor.optimize).lower() == 'minimize':
            reverse = False

        fig, ax = plt.subplots()

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
            population_fitness.append(
                sum(already_checked[:self.instructor.population]) / self.instructor.population
            )
            if show_kids:
                current_gen = [g for _ in all_fit]
                ax.scatter(current_gen, all_fit, marker='o', edgecolors='lightgray',
                           facecolors='none', alpha=0.5, s=3)
        print('-------------')
        print('Generations:', avail_gens)
        print('Population fitness:', population_fitness)
        print('-------------')

        ax.plot(avail_gens, population_fitness, color='black', linestyle='-',
                label='Population', linewidth=2)
        ax.plot(avail_gens, ave_kids, color='green', linestyle='-',
                label='Av. fitness', linewidth=1)
        ax.plot(avail_gens, best_kid, color='gray', linestyle='--',
                label='Best fitness', linewidth=1)
        ax.plot(avail_gens, worst_kid, color='gray', linestyle='-.',
                label='Worst fitness', linewidth=1)

        ax.set_xlabel('Generations')
        ax.set_ylabel('Fitness')

        if show_std:
            ax2 = ax.twinx()
            ax2.plot(avail_gens, std_kids, color='skyblue', linestyle=':',
                     label='Std. dev.', linewidth=1)
            ax2.set_ylabel('Std. Dev.')
            lines_1, labels_1 = ax.get_legend_handles_labels()
            line_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(lines_1 + line_2, labels_1 + labels_2, loc='best')
        else:
            ax.legend(loc='best')

        fig.tight_layout()
        plt.show()
        fig.savefig('evolution.png', dpi=300)

    # validators ------------------------------------------------------
    def is_valid_sequence(self, seq) -> bool:
        """
        Evolver-level validity: length, duplicates, prohibited patterns and
        exclusions. Physicochemical restrictions are enforced by the Generator
        (Restriction objects), not here.
        """
        seq = str(seq)
        logger.debug(f'Evolver: checking validity of sequence {seq}')

        if self.sequence_exists(seq) and self.instructor.avoid_reinsertion:
            logger.debug('Evolver: sequence already exists --> discarding')
            return False

        validity_value = (
            not self.has_prohibited_pattern(seq)
            and not self.is_excluded_sequence(seq)
            and self.fits_length(seq)
        )
        logger.debug(f'Evolver: is sequence {seq} valid? : {validity_value}')
        return validity_value

    def is_excluded_sequence(self, seq) -> bool:
        """True if the sequence should be excluded."""
        return str(seq) in self.excluded_sequences

    def has_prohibited_pattern(self, seq) -> bool:
        """True if a prohibited (plain) pattern is found."""
        seq = str(seq)
        patterns_clean = [k for k in self.prohibited_patterns if '*' not in k]
        return any(pat in seq for pat in patterns_clean)

    def sequence_exists(self, seq) -> bool:
        """True if the sequence string already exists in any list."""
        seq = str(seq)
        all_sequences = self.sequences + self.parent_sequences + self.discarded_sequences
        return seq in [str(s) for s in all_sequences]

    def fits_length(self, seq):
        """Checks if the sequence matches instructor.peptide_len."""
        return len(seq) == self.instructor.peptide_len

    # decide and choose ----------------------------------------------------
    def is_top_index(self, index):
        """True if the index is part of the elite section."""
        num_indexes = int(self.instructor.population * self.instructor.elite_ratio)
        return index in range(num_indexes)

    def take_bool_decision(self, probability=0.1):
        """Take a boolean decision based on the probability."""
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability should be between 0.0 and 1.0")
        return random.random() < probability

    def choose_sequence(
            self, weighted=False, reverse=False, exception=None, include_elite=True,
            include_discarded=False, include_current=False,
            only_discarded=False, only_current=False,
    ):
        """Return a Sequence object chosen randomly, optionally weighted."""
        population = self.parent_sequences

        if include_discarded:
            population = population + self.discarded_sequences
        if include_current:
            population = population + self.sequences

        if only_discarded:
            population = self.discarded_sequences
        if only_current:
            population = self.sequences

        if exception:
            if not isinstance(exception, (list, tuple, set)):
                exception = [exception]
            population = [k for k in population if k not in exception]
        if not include_elite:
            population = [k for k in population if not k.is_elite]

        if not population:
            return None

        if not weighted:
            return random.choice(population)

        if reverse:
            population = population[::-1]

        weights = []
        n = len(population)
        for idx, seq in enumerate(population):
            base_weight = (n - idx) * self.instructor.weight_bias
            if seq.is_elite:
                multiplier = self.instructor.elite_bias
                if reverse:
                    try:
                        multiplier = 1 / multiplier
                    except ZeroDivisionError:
                        multiplier = 0.0
                base_weight *= multiplier
            weights.append(base_weight)
        return random.choices(population, weights=weights, k=1)[0]

    # candidate generation -------------------------------------------------
    def _choose_parents(self):
        """
        Choose up to two parents for the Generator. Returns a (parent1, parent2)
        tuple. Either entry may be None when not enough sequences are available;
        the Generator decides what to do with the parents it receives.
        """
        parent1 = self.choose_sequence(
            weighted=self.instructor.populate_weighted,
            include_elite=True,
            include_discarded=self.instructor.include_discarded,
        )
        parent2 = self.choose_sequence(
            weighted=self.instructor.populate_weighted,
            include_elite=True,
            exception=parent1,
            include_discarded=self.instructor.include_discarded,
        )
        return parent1, parent2

    def _make_candidate(self, first=False):
        """
        Produce a single valid Sequence using the Generator.

        On the first population (or when there are fewer than two parents)
        no parents are passed, so the Generator builds from scratch. The
        Generator already enforces all Restriction objects; Evolver only adds
        length / duplicate / exclusion checks and handles reinsertion.
        """
        while True:
            if first:
                parent1, parent2 = None, None
            else:
                parent1, parent2 = self._choose_parents()

            candidate = self.generator.generate(parent1, parent2)

            if self.is_valid_sequence(candidate):
                break

        # reinsertion of a previously generated sequence
        if self.sequence_exists(candidate):
            candidate = self.take_sequence(candidate)
            candidate.check_reinsertion(iterations_preferent=self.instructor.iterations_elite)
        else:
            candidate = Sequence(
                candidate,
                generation=self.generations,
                h_scale=self.instructor.hydrophobic_scale,
            )
        return candidate

    # first sequences ------------------------------------------------------
    def first_sequences(self) -> None:
        logger.info('Evolver: Starting sequences')

        if len(self.instructor.prohibited_patterns) > 0:
            self.prohibited_patterns.extend(self.instructor.prohibited_patterns)

        if len(self.instructor.excluded_sequences) > 0:
            self.excluded_sequences.extend(self.instructor.excluded_sequences)

        if len(self.instructor.sequences) > 0:
            if len(self.instructor.sequences) > self.instructor.population:
                logger.warning(
                    f'Evolver: More than {self.instructor.population} seed sequences '
                    f'--> taking the first {self.instructor.population}'
                )
            for sq in self.instructor.sequences[:self.instructor.population]:
                if len(sq) != self.instructor.peptide_len:
                    logger.warning(
                        f'Evolver: Length does not match "{sq}" '
                        f'(expected {self.instructor.peptide_len}) --> skipping'
                    )
                    continue
                if self.instructor.check_validity:
                    if self.is_excluded_sequence(sq):
                        logger.warning(f'Evolver: Sequence "{sq}" is excluded --> skipping')
                        continue
                    if self.has_prohibited_pattern(sq):
                        logger.warning(f'Evolver: Sequence "{sq}" has a prohibited pattern --> skipping')
                        continue
                    if self.sequence_exists(sq):
                        logger.warning(f'Evolver: Sequence "{sq}" already exists --> skipping')
                        continue
                self.sequences.append(
                    Sequence(sq, h_scale=self.instructor.hydrophobic_scale)
                )

    def include_sequences(self):
        for candidate in self.to_include:
            if len(self.sequences) == self.instructor.population:
                logger.warning('Evolver: evolver is full --> skipping external insertion')
                break
            if len(candidate) != self.instructor.peptide_len:
                logger.warning(
                    f'Evolver: Length does not match "{candidate}" '
                    f'(expected {self.instructor.peptide_len}) --> not included'
                )
                continue
            if not self.is_valid_sequence(candidate):
                logger.warning(f'Evolver: Sequence {candidate} is not valid --> not included')
                continue
            if self.sequence_exists(candidate):
                logger.warning(f'Evolver: Sequence {candidate} was previously generated --> reinserting')
                candidate = self.take_sequence(candidate)
                candidate.check_reinsertion(iterations_preferent=self.instructor.iterations_elite)
            else:
                candidate = Sequence(
                    candidate,
                    generation=self.generations,
                    h_scale=self.instructor.hydrophobic_scale,
                )
            self.sequences.append(candidate)
        self.to_include = []

    # populate -------------------------------------------------------------
    def populate(self) -> None:
        """Populate self.sequences up to instructor.population."""
        if len(self.sequences) == self.instructor.population:
            logger.warning('Evolver: Already populated --> skipping')
            return

        if self.started and len(self.sequences) < self.instructor.population:
            self.generations += 1

        if not self.started:
            # first population: build from scratch
            logger.info('Evolver: First population (random fill from Generator)')
            while len(self.sequences) < self.instructor.population:
                candidate = self._make_candidate(first=True)
                logger.info(f'New sequence: {candidate}')
                self.sequences.append(candidate)
        else:
            # external sequences first
            try:
                len(self.to_include)
            except TypeError:
                self.to_include = []
            if len(self.to_include) > 0:
                logger.info('Evolver: External sequences found --> including sequences')
                self.include_sequences()

            logger.info(f'Evolver: Populating with methods {self.instructor.methods}')
            while len(self.sequences) < self.instructor.population:
                candidate = self._make_candidate(first=False)
                logger.info(f'New sequence: {candidate}')
                self.sequences.append(candidate)

            self.discarded_sequences = self.parent_sequences + self.discarded_sequences
            self.parent_sequences = []

    # methods to save and restore sequences ---------------------------------
    def read_previous(self):
        """
        Read sequences from previous simulations located in
        instructor.evomd_directory and run only the analysis step.
        """
        import os
        self.started = True
        logger.warning('Evolver: removing sequences list to include previous calculations')
        self.sequences = []
        sequences = os.listdir(self.instructor.evomd_directory)
        for seq in sequences:
            if not self.fits_length(seq):
                continue
            if not self.is_valid_sequence(seq) and self.instructor.check_validity:
                continue
            new_seq = Sequence(seq, h_scale=self.instructor.hydrophobic_scale)
            new_seq.has_directory = True
            new_seq.directory = os.path.join(
                self.instructor.cwd, self.instructor.evomd_directory, seq
            )
            new_seq.last_iter_dir = os.path.join(new_seq.directory, 'iter_1')
            new_seq.is_waiting_analysis = True
            self.sequences.append(new_seq)
        self.iterate(new_plan=[False, False, False, True])
        self.sequences = [
            k for k in self.sequences
            if k.fitness is not None and not math.isnan(k.fitness)
        ]
        self.sort_sequences()
        logger.info(f'Evolver: {len(self.sequences)} in sequences list')
        logger.info(f'Evolver: {len(self.parent_sequences)} in parent sequences list')
        logger.info(f'Evolver: {len(self.discarded_sequences)} in discarded sequences list')

    def sequence_backup(self):
        """Create a sequence.json file in each Sequence.directory."""
        from utils import save_json
        import os
        logger.info('Evolver: Starting sequence back up . . .')
        all_sequences = self.discarded_sequences + self.sequences + self.parent_sequences
        for seq in all_sequences:
            if not seq.has_directory:
                logger.debug(
                    f'Evolver: sequence_backup found a sequence without directory: '
                    f'{str(seq)} --> creating directory'
                )
                self.manager.create_sequence_directory(seq)
            outfile = os.path.join(seq.directory, 'sequence.json')
            save_json(seq, outfile)
        logger.info('Evolver: Sequence back up is done')

    # sorting and moving ----------------------------------------------------
    def sort_sequences(self) -> None:
        """
        Order sequences according to instructor.optimize and split them into
        self.sequences (elite), self.discarded_sequences (worst) and
        self.parent_sequences (held for populating).
        """
        logger.info('Evolver: Sorting sequences')
        all_sequences = self.sequences + self.discarded_sequences

        for seq in all_sequences:
            seq.is_elite = False
            seq.is_top = False
            seq.is_discarded = True
            seq.current_index = None

        valid = [s for s in all_sequences
                 if s.fitness is not None and not math.isnan(s.fitness)]
        invalid = [s for s in all_sequences
                   if s.fitness is None or math.isnan(s.fitness)]

        reverse = True  # maximize is default
        if str(self.instructor.optimize).lower() == 'minimize':
            reverse = False

        valid.sort(key=lambda seq: seq.fitness, reverse=reverse)
        all_sequences = valid + invalid

        num_total = self.instructor.population
        num_elite = int(num_total * self.instructor.elite_ratio)
        num_to_keep = num_total - int(num_total * self.instructor.discard_ratio)

        self.sequences = []
        self.parent_sequences = []
        self.discarded_sequences = []

        for idx, seq in enumerate(all_sequences):
            seq.current_index = idx

            if idx < num_elite:
                seq.is_top = True

            if self.generations > self.instructor.iterations_elite:
                seq.check_elite()

            if seq.is_elite:
                self.sequences.append(seq)
                seq.is_discarded = False
            elif len(self.parent_sequences) < (num_to_keep - len(self.sequences)):
                self.parent_sequences.append(seq)
                seq.is_discarded = False
            else:
                self.discarded_sequences.append(seq)

        logger.debug(
            f'Evolver.sort_sequences(): sequences {len(self.sequences)} '
            f'parent {len(self.parent_sequences)} discarded {len(self.discarded_sequences)}'
        )

    def set_failed(self):
        """Move sequences with is_failed=True to self.failed_sequences."""
        not_failed = [k for k in self.sequences if not k.is_failed]
        failed = [k for k in self.sequences if k.is_failed]
        if len(failed) > 0:
            logger.warning(f'Evolver: Failed sequences found: {len(failed)}')
        self.sequences = not_failed
        self.failed_sequences = failed

    def take_sequence(self, seq):
        """Find a sequence in any list, remove it and return the object."""
        seq = str(seq)
        for seq_list in (self.discarded_sequences, self.parent_sequences, self.sequences):
            for old in seq_list:
                if str(old) == seq:
                    seq_list.remove(old)
                    logger.info(f'Evolver: take_sequence: sequence {seq} is taken')
                    return old
        logger.error(
            f'Evolver: take_sequence: trying to take {seq} but it was not found '
            f'in any list --> stopping evolution'
        )
        exit(2)

    # Iterate ----------------------------------------------------
    def is_valid_plan(self, plan):
        if not isinstance(plan, list):
            logger.warning('Evolver: new plan must be a list')
            return False
        if len(plan) != 4:
            logger.warning('Evolver: new plan must contain 4 items')
            return False
        if not all(isinstance(k, bool) for k in plan):
            logger.warning('Evolver: new plan must contain only bool values')
            return False
        return True

    def iterate(self, new_plan=None):
        """
        Perform one full iteration of the evolutionary cycle:
        construct systems, run simulations, check status, analyze results
        and mark failed simulations.
        """
        logger.info("Evolver: Starting iteration step")

        step_flags = {
            'construct': True,
            'calculate': True,
            'check': True,
            'analyze': True,
        }

        if self.recover_enabled:
            logger.info("Evolver: Recovery flag set. Attempting to resume previous session...")
            step_flags = self.manager.recover_pending_sequences(step_flags)
            self.recover_enabled = False
        elif new_plan:
            logger.info('Evolver: new iteration plan found --> trying to set new plan')
            if self.is_valid_plan(new_plan):
                for step, new in zip(step_flags, new_plan):
                    step_flags[step] = new

        logger.info(f'Evolver: iteration plan: {step_flags}')

        if step_flags['construct']:
            logger.info("Evolver: Running constructor step")
            try:
                self.manager.run_constructors()
            except Exception as e:
                logger.error(f"Evolver: Constructor step failed with error: {e}")
                return
            self.save_pkl()

        if step_flags['calculate']:
            logger.info("Evolver: Running calculator step")
            try:
                self.manager.run_calculators()
            except Exception as e:
                logger.error(f"Evolver: Calculator step failed with error: {e}")
                return
            self.save_pkl()

        if step_flags['check']:
            logger.info("Evolver: Checking simulation status")
            try:
                self.manager.run_checkers()
            except Exception as e:
                logger.error(f"Evolver: Checker step failed with error: {e}")
                return
            self.save_pkl()

        if step_flags['analyze']:
            logger.info("Evolver: Running analyzer step")
            try:
                self.manager.run_analyzer()
            except Exception as e:
                logger.error(f"Evolver: Analyzer step failed with error: {e}")
                return
            self.save_pkl()

        self.set_failed()
        logger.info("Evolver: Iteration step completed")

    # Convergence or max generations? -----------------------------
    def check_termination(self):
        """Evaluate termination conditions and set self.runnable = False if met."""
        logger.info("Evolver: Checking termination criteria")
        if self.generations >= 1000:
            logger.info("Evolver: Max generations reached --> stopping")
            self.runnable = False


if __name__ == '__main__':
    pass

# === manager.py ===
"""
Manager: orchestrates the per-sequence simulation lifecycle for the Evolver.

The Manager is the bridge between the Evolver and the user-supplied external
modules (constructor / calculator / analyzer, named in the Instructor). For
each sequence in the Evolver's current population, it runs the iteration steps
in order: construct the simulation system, launch the calculation, poll until
it finishes, and analyze the result into a fitness value.

Responsibilities:
  - Directory layout: one directory per sequence under evomd_directory, with a
    numbered iter_N subdirectory per simulation attempt.
  - Method dispatch: each external function is looked up by name from its module
    and executed inside the sequence's iteration directory.
  - State bookkeeping: sets the Sequence flags (is_just_constructed, is_running,
    is_waiting_analysis, is_failed) and counters as each step progresses.
  - Recovery: on a restarted session, inspects sequence state to decide which
    steps can be skipped.

The external functions follow a fixed interface (constructor_method,
calculator_method, calculator_check, analyzer_method), each taking the Sequence
as `sequence=` and, where relevant, returning a value (readiness bool / fitness
float).
"""

import os
import logging
from .utils import get_module_function, current_time
from contextlib import contextmanager
import time
from .exceptions_evomd import MethodFailedError, MethodExistError, EmptyPopulationError


logger = logging.getLogger(__name__)


class Manager:
    """
    Runs the construct/calculate/check/analyze cycle over the Evolver's
    sequences, managing their directories and external-method dispatch.

    Class attributes:
        iter_dir_prefix (str): Prefix for per-attempt iteration subdirectories.
        constructor_name / calculator_name / calculator_check / analyzer_name  (str): Names of the functions looked up in the user modules.
    """

    name = 'manager'
    iter_dir_prefix = 'iter_'
    constructor_name = 'constructor_method'
    calculator_name = 'calculator_method'
    calculator_check = 'calculator_check'
    analyzer_name = 'analyzer_method'

    def __init__(self, evolver):
        """
        Bind the Manager to its Evolver and ensure the base directory exists.

        Args:
            evolver (Evolver): The owning Evolver; used to reach the Instructor
                configuration and the current sequence lists.
        """
        self.evolver = evolver
        self.base_dir = os.path.join(
            self.evolver.instructor.cwd, 
            self.evolver.instructor.evomd_directory
            )
        os.makedirs(self.base_dir, exist_ok=True)

    # create directories and manage them ---------------------
    @contextmanager
    def working_directory(self, path):
        """Temporarily chdir into `path`, restoring the previous cwd on exit."""
        prev_cwd = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(prev_cwd)

    def create_sequence_directory(self, seq) -> None:
        """Create the base directory for a sequence and record it on the object."""
        directory = os.path.join(self.base_dir, str(seq))
        os.makedirs(directory, exist_ok=True)
        logger.debug(f'Manager: Directory created for sequence {str(seq)}')
        seq.has_directory = True
        seq.directory = directory

    def create_iteration_directory(self, seq) -> None:
        """
        Create the iter_N subdirectory for the sequence's next attempt.

        Ensures the sequence has a base directory first, then creates the
        attempt directory (numbered from simulation_attempts + 1) and stores it
        as seq.last_iter_dir.
        """
        if not seq.has_directory:
            logger.debug(f'Manager: Sequence without directory: {str(seq)} --> creating directory')
            self.create_sequence_directory(seq)
        directory = os.path.join(seq.directory, f'{self.iter_dir_prefix}{seq.simulation_attempts+1}')
        os.makedirs(directory, exist_ok=True)
        seq.last_iter_dir = directory

    # execute functions -----------------  before these, execute create_iteration_directory()
    def execute_method(self, seq, method, return_value=False) -> None:
        """
        Run an external method for a sequence inside its iteration directory.

        Args:
            seq (Sequence): The sequence passed to the method as `sequence=`.
            method (callable): The external function to run.
            return_value (bool): If True, return the method's result; otherwise
                run it for its side effects only.

        Returns:
            The method's return value when return_value is True, else returns None.
        """
        seq_id = str(seq)
        with self.working_directory(seq.last_iter_dir):
            logger.info(f"Manager: execute_method: sequence '{seq_id}', directory '{seq.last_iter_dir}'")
            if return_value:
                return method(sequence=seq)
            else:
                method(sequence=seq)

    # run codes ---------------------
    def run_constructors(self):
        """
        Run the constructor for every current sequence.

        Looks up constructor_method in the configured constructor module, then
        for each sequence creates its iteration directory, runs the constructor
        there, and marks it is_just_constructed. Raises MethodExistError if no 
        constructor module is configured and EmptyPopulationError if Evolver.sequences
        is empty; per-sequence errors are logged and counted in exceptions. 
        If exceptions > 0 raises MethodFailedError with code=exceptions.
        """
        constructor_module = self.evolver.instructor.constructor
        if not constructor_module:
            logger.error("Manager: No constructor module defined in Instructor.")
            raise MethodExistError("Constructor method was not found.")
        if len(self.evolver.sequences) == 0:
            raise EmptyPopulationError("Evolver.sequences is empty. Nothing to construct.")
        # get function from constructor module
        constructor_function = get_module_function(constructor_module, self.constructor_name, critical=True)
        # iterate on sequences
        exceptions = 0
        for seq in self.evolver.sequences:
            try:
                # be sure that Sequence.last_iter_dir is defined
                self.create_iteration_directory(seq)
                # execute in Sequence.last_iter_dir
                self.execute_method(seq, constructor_function)
                # set label
                seq.is_just_constructed = True
            except Exception as e:
                logger.error(f"Manager: Error while running {self.constructor_name} for sequence '{seq}': {e}")
                exceptions += 1
                continue
        
        if exceptions > 0:
            raise MethodFailedError("Something went wrong while constructing sequences.", code=exceptions)
        logger.info(f'Manager: Ending run_constructors function.')

    def run_calculators(self):
        """
        Launch the calculation for every current sequence.

        Looks up calculator_method in the configured calculator module. For each
        sequence, increments simulation_attempts, runs the calculator, and marks
        it is_running (and no longer just-constructed). 
        Raises MethodExistError if no calculator module is configured and 
        EmptyPopulationError if Evolver.sequences is empty; 
        per-sequence errors are logged and counted in exceptions. 
        If exceptions > 0 raises MethodFailedError
        with code=exceptions.
        """
        logger.info(f'Manager: run_calculators: {current_time()}')
        calculator_module = self.evolver.instructor.calculator
        if not calculator_module:
            logger.error("Manager: No calculator module defined in Instructor.")
            raise MethodExistError("Calculator method was not found.")
        if len(self.evolver.sequences) == 0:
            raise EmptyPopulationError("Evolver.sequences is empty. Nothing to calculate.")
        # get the function
        calculator_function = get_module_function(calculator_module, self.calculator_name, critical=True)
        # iterate on sequences
        # store number of errors to verify if all the sequences are running
        exceptions = 0
        for seq in self.evolver.sequences:
            seq.simulation_attempts += 1
            try:
                # execute
                self.execute_method(seq, calculator_function)
                # set labels
                seq.is_just_constructed = False
                seq.is_running = True   # sequences is running
                logger.info(f'Sequence {str(seq)} is being simulated')
            except Exception as e:
                logger.error(f"Manager: Error while running {self.calculator_name} for sequence '{seq}': {e}")
                exceptions += 1
                continue
        
        # did all the sequences fail?
        if exceptions > 0:
            raise MethodFailedError("Execution of calculator method failed for all the sequences.", code=exceptions)
        logger.info(f'Manager: Ending run_calculators function.')
    
    def run_checkers(self):
        """
        Poll running simulations until they finish or the cycle budget is spent.

        Looks up calculator_check in the calculator module and repeatedly calls
        it for each running sequence. A sequence reported ready is marked
        is_running=False, is_waiting_analysis=True. Between rounds the Manager
        sleeps for instructor.sleep_time. Once instructor.max_check_cycle is
        reached, any sequence not yet waiting for analysis is marked failed.
        Raises MethodExistError if no calculator_check module is configured.
        """
        logger.info(f'Manager: run_checkers: {current_time()}')
        calculator_module = self.evolver.instructor.calculator_check
        if not calculator_module:
            logger.error("Manager: No calculator_check module defined in Instructor.")
            raise MethodExistError("Calculator check method was not found.")
        # get function
        logger.debug('Manager: calculator_check value: {}'.format(self.calculator_check))
        calculator_check_function = get_module_function(calculator_module, self.calculator_check, critical=True)
        # set to False: True when all calculations are ready
        
        logger.debug('Manager: looking for running sequences')
        # All entries must become True for the polling loop to stop.
        seq_ready_status = [not s.is_running for s in self.evolver.sequences]
        logger.debug('Manager: Running sequences {}'.format(len(seq_ready_status)))

        # loop to check computations several times
        # time.sleep(self.evolver.instructor.sleep_time)
        check_cycles = 0
        while not all(seq_ready_status):
            # show time
            logger.info(f'Manager: checking calculations {current_time()}: cycle {check_cycles}/{self.evolver.instructor.max_check_cycle}')
            # iterate on sequences 
            for idx, seq in enumerate(self.evolver.sequences):
                try:
                    ready = self.execute_method(seq, calculator_check_function, return_value=True)
                    seq_ready_status[idx] = ready
                    if ready:
                        seq.is_running = False
                        seq.is_waiting_analysis = True
                except Exception as e:
                    logger.error(f"Manager: Error checking sequence '{seq}': {e}")
                    continue
            # if not all ready yet and budget remains, wait before re-polling
            if not all(seq_ready_status) and check_cycles < self.evolver.instructor.max_check_cycle:
                sleep_time = self.evolver.instructor.sleep_time
                logger.info(f"Manager: Waiting for {sleep_time} seconds before rechecking.")
                time.sleep(sleep_time)
                check_cycles += 1
            # budget spent: mark every still-unfinished sequence as failed
            if check_cycles >= self.evolver.instructor.max_check_cycle:
                logger.warning(f"Manager: Maximun checking cycles ({self.evolver.instructor.max_check_cycle}) exceeded.")
                for seq in self.evolver.sequences:
                    if seq.is_waiting_analysis:
                        continue
                    seq.failed_simulations += 1
                    seq.is_running = False
                    seq.is_failed = True
                break
        # Stop loop 
        logger.info("Manager: Ending run_checkers function.")

    def run_analyzer(self):
        """
        Analyze finished simulations into fitness values.

        Looks up analyzer_method in the analyzer module. For each sequence
        waiting for analysis, runs the analyzer, appends the returned value to
        its fitness_list, clears is_waiting_analysis, and increments
        completed_simulations. Raises MethodExistError if no analyzer module 
        is configured; per-sequence errors are logged and skipped.
        """
        logger.info(f'Manager: run_analyzer: {current_time()}')
        # get analyzer module
        analyzer_module = self.evolver.instructor.analyzer
        if not analyzer_module:
            logger.error("Manager: No analyzer module defined in Instructor.")
            raise MethodExistError("Analyzer method was not found.")
        # get function
        analyzer_function = get_module_function(analyzer_module, self.analyzer_name, critical=True)

        # iterate on sequences
        for seq in self.evolver.sequences:
            if not seq.is_waiting_analysis:
                continue
            try:
                fitness_value = self.execute_method(seq, analyzer_function, return_value=True)
                logger.info(f'Manager: Sequence {str(seq)}, value {fitness_value}')
                seq.fitness_list.append(fitness_value)
                seq.is_waiting_analysis = False
                seq.completed_simulations += 1

            except Exception as e:
                logger.error(f"Manager: Error while running {self.analyzer_name} for sequence '{seq}' in directory '{seq.last_iter_dir}': {e}")
                continue   # continue with the next sequence
        logger.info('Manager: Ending run_analyzer function')

    # recover methods ----------------------------------------------
    def recover_pending_sequences(self, step_flags: dict) -> dict:
        """
        Inspect sequence state to decide which iteration steps to re-run.

        On a restarted session, examines each sequence's flags to find the
        furthest-along pending step and disables steps already completed. If the
        population is incomplete, all steps are disabled (nothing to resume).

        Args:
            step_flags (dict): Initial step plan (construct/calculate/check/analyze).

        Returns:
            dict: The adjusted step plan after inspecting sequence states.
        """
        logger.info("Manager: Recovering pending sequences from last session...")
        sequences = self.evolver.sequences

        if len(sequences) < self.evolver.instructor.population:
            logger.info('Manager: Not enough population --> skipping iteration')
            step_flags = {
                'construct': False,
                'calculate': False,
                'check': False,
                'analyze': False,
            }
            return step_flags

        need_calculate = []
        need_check = []
        need_analyze = []
        for seq in sequences:
            # constructed but not yet calculated --> needs the calculate step
            try:  # guarded: is_just_constructed was added to Sequence later
                need_calculate.append(seq.is_just_constructed)
            except:
                need_calculate.append(False)

            # currently running --> needs the check step
            need_check.append(seq.is_running)

            # finished but not analyzed --> needs the analyze step
            need_analyze.append(seq.is_waiting_analysis)

        # Enable each step if it (or any earlier pending step) has work to do.
        step_flags = {
            'construct': False,
            'calculate': any(need_calculate),
            'check': any([any(need_check), any(need_calculate)]),
            'analyze': any([any(need_analyze), any(need_check), any(need_calculate)]),
        }

        logger.info(f"Manager: Step plan after recovery inspection: {step_flags}")
        return step_flags
    

if __name__ == '__main__':
    pass

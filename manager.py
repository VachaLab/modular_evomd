# === manager.py ===
import os
import logging
from utils import get_module_function, current_time
from contextlib import contextmanager
import time


logger = logging.getLogger(__name__)


class Manager:
    name = 'manager'
    iter_dir_prefix = 'iter_'
    contructor_name = 'constructor_method'
    calculator_name = 'calculator_method'
    calculator_check = 'calculator_check'
    analyzer_name = 'analyzer_method'
    penalty_name = 'penalty_method'

    def __init__(self, evolver):
        self.evolver = evolver
        self.base_dir = os.path.join(
            self.evolver.instructor.cwd, 
            self.evolver.instructor.evomd_directory
            )
        os.makedirs(self.base_dir, exist_ok=True)

    # create directories and manage them ---------------------
    @contextmanager
    def working_directory(self, path):
        """Context manager to temporarily change the working directory."""
        prev_cwd = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(prev_cwd)

    def create_sequence_directory(self, seq) -> None:
        directory = os.path.join(self.base_dir, str(seq))
        os.makedirs(directory, exist_ok=True)
        logger.debug(f'Manager: Directory created for sequence {str(seq)}')
        seq.has_directory = True
        seq.directory = directory

    def create_iteration_directory(self, seq) -> None:
        if not seq.has_directory:
            logger.debug(f'Manager: Sequence without directory: {str(seq)} --> creating directory')
            self.create_sequence_directory(seq)
        directory = os.path.join(seq.directory, f'{self.iter_dir_prefix}{seq.simulation_attempts+1}')
        os.makedirs(directory, exist_ok=True)
        seq.last_iter_dir = directory

    # execute functions -----------------  before these, execute create_iteration_directory()
    def execute_method(self, seq, method, return_value=False) -> None:
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
        creates iteration directory and sets Sequence.last_iter_dir
        """
        constructor_module = self.evolver.instructor.constructor
        if not constructor_module:
            logger.error("Manager: No constructor module defined in Instructor --> exit")
            exit(2)
        # get function from constructor module
        constructor_function = get_module_function(constructor_module, self.contructor_name, critical=True)
        # iterate on sequences
        for seq in self.evolver.sequences:
            try:
                # be sure that Sequence.last_iter_dir is defined
                self.create_iteration_directory(seq)
                # execute in Sequence.last_iter_dir
                self.execute_method(seq, constructor_function)
                # set label
                seq.is_just_constructed = True
            except Exception as e:
                logger.error(f"Manager: Error while running {self.contructor_name} for sequence '{seq}': {e}")
                continue   # continue with the next sequence

    def run_calculators(self):
        """
        Runs calculations in external software and sets 
        Sequence.is_running = True
        and Sequence.simulation_attempts += 1
        """
        logger.info(f'Manager: run_calculators: {current_time()}')
        calculator_module = self.evolver.instructor.calculator
        if not calculator_module:
            logger.error("Manager: No calculator module defined in Instructor --> exit")
            exit(2)
        
        # get the function
        calculator_function = get_module_function(calculator_module, self.calculator_name, critical=True)
        # iterate on sequences
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
                continue
        logger.info(f'Manager: Ending run_calculators function')
    
    def run_checkers(self):
        """
        Runs checkers to identify finished simulaltions and sets
        Sequence.is_running = False
        Sequence.is_waiting_analysis = True / False
        Sequence.failed_simulations += 1 if simulation failed
        Sequence.is_failed = True
        """
        logger.info(f'Manager: run_checkers: {current_time()}')
        calculator_module = self.evolver.instructor.calculator
        if not calculator_module:
            logger.error("Manager: No calculator module defined in Instructor --> exit")
            exit(2)
        # get function
        logger.debug('Manager: calculator_check value: {}'.format(self.calculator_check))
        calculator_check_function = get_module_function(calculator_module, self.calculator_check, critical=True)
        # set to False: True when all calculations are ready
        
        logger.debug('Manager: looking for running sequences')
        seq_ready_status = [not s.is_running for s in self.evolver.sequences]  # this list must be all true to stop while.
        logger.debug('Manager: Running sequences {}'.format(len(seq_ready_status)))

        # loop to check computations several times
        # time.sleep(self.evolver.instructor.sleep_time)
        check_cycles = 0
        while not all(seq_ready_status):
            # show time
            logger.info(f'Manager: checking calculations {current_time()}: cycle {check_cycles}/{self.evolver.instructor.max_check_cycle}')
            # iterate on sequences 
            for idx, seq in enumerate(self.evolver.sequences):
                # if not seq.is_running:
                #     seq_ready_status[idx] = True
                #     continue
                # exeucte calculator_check
                try:
                    ready = self.execute_method(seq, calculator_check_function, return_value=True)
                    seq_ready_status[idx] = ready
                    if ready:
                        seq.is_running = False
                        seq.is_waiting_analysis = True
                except Exception as e:
                    logger.error(f"Manager: Error checking sequence '{seq}': {e}")
                    continue
            # if not all_ready, wait
            if not all(seq_ready_status) and check_cycles < self.evolver.instructor.max_check_cycle:
                # Si no todos los cálculos están listos, espera antes de volver a intentar
                sleep_time = self.evolver.instructor.sleep_time  # Obtiene el tiempo de espera desde Instructor
                logger.info(f"Manager: Waiting for {sleep_time} seconds before rechecking.")
                time.sleep(sleep_time)
                check_cycles += 1
            if check_cycles > self.evolver.instructor.max_check_cycle:
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
        Analyze simulations if Sequence.is_waiting_analysis = True
        Sets:
        Sequence.is_waiting_analysis = False
        Sequence.completed_simulations += 1

        """
        logger.info(f'Manager: run_analyzer: {current_time()}')
        # get analyzer module
        analyzer_module = self.evolver.instructor.analyzer
        if not analyzer_module:
            logger.error("Manager: No analyzer module defined in Instructor --> exit")
            exit(2)
        # get function
        analyzer_function = get_module_function(analyzer_module, self.analyzer_name, critical=True)

        # iterate on sequences
        for seq in self.evolver.sequences:
            if not seq.is_waiting_analysis:
                continue
            try:
                fitness_value = self.execute_method(seq, analyzer_function, return_value=True)
                logger.info(f'Manager: Sequence {str(seq)}, value {fitness_value}')
                seq.fitness.append(fitness_value)
                seq.is_waiting_analysis = False
                seq.completed_simulations += 1

            except Exception as e:
                logger.error(f"Manager: Error while running {self.analyzer_name} for sequence '{seq}' in directory '{seq.last_iter_dir}': {e}")
                continue   # continue with the next sequence
        logger.info('Manager: Ending run_analyzer function')

    # recover methods ----------------------------------------------
    def recover_pending_sequences(self, step_flags: dict) -> dict:
        """
        Recover state of sequences and update step_flags dict to skip already completed steps.

        Args:
            step_flags (dict): Dictionary controlling which steps are still needed.

        Returns:
            dict: Updated step_flags after inspecting sequence states.
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
            # is it just constructed? --> run simulations
            try:  # try this, since Sequence.is_just_constructed was defined a posteriori
                need_calculate.append(seq.is_just_constructed)
            except:
                need_calculate.append(False)

            # is it runnig simulations? --> check simulations
            need_check.append(seq.is_running)

            # is it waiting for analysis? --> analysis
            need_analyze.append(seq.is_waiting_analysis)

        # shut down redundant steps

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

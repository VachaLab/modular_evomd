# === sequence.py ===
import numpy as np
from residue import Residue
import logging

logger = logging.getLogger(__name__)

class Sequence:
    name = 'sequence'

    def __init__(self, seq, generation=0, h_scale='eisenberg') -> None:
        self.sequence = seq
        self.hydrophobic_scale = h_scale
        self.residues = [Residue(k, n, self.hydrophobic_scale) for n, k in enumerate(self.sequence)]
        self.generation = generation  # in which generation was created
        self.fitness = []
        self.penalties = []
        # --- properties ---
        self.hydrophobic_moment, self.hydrophobic_vector, self.raw_positions = self.compute_hydrophobic_moment()
        self.charge = self.compute_charge()
        self.n_ter_charge = self.residues[0].charge
        self.c_ter_charge = self.residues[-1].charge
        # --- boolean info ---
        self.is_elite = False
        self.is_preferent = False
        self.is_discarded = False
        self.is_top = False  # true if its index is in elite region
        self.is_resurrected = False
        self.is_just_constructed = False
        self.is_running = False
        self.is_waiting_analysis = False
        self.is_failed = False
        self.has_directory = False
        # --- counting ---
        self.simulation_attempts = 0
        self.completed_simulations = 0
        self.failed_simulations = 0
        self.reinsertions = 0  # reinserted into Evolver.sequences by generator
        self.resurrections = 0  # reinserted into Evolver.sequences by resurrection
        self.times_elite = 0
        # --- index and directories ---
        self.current_index = None
        self.directory = None
        self.last_iter_dir = None
    
    # special methods -------------------------------
    def __str__(self):
        return self.sequence
    
    def __len__(self):
        return len(self.sequence)
    
    def __iter__(self):
        return iter(self.sequence)
    
    # properties -----------------------------------
    def compute_charge(self):
        charges = [k.charge for k in self.residues]
        return sum(charges)

    def compute_hydrophobic_moment(self, scale=None, theta=100):
        # get contributions to hydrophobic vecotr
        h_vector = np.zeros(3)
        for res in self.residues:
            h_vector += res.get_hm_contribution()
        h_scalar = np.linalg.norm(h_vector)
        seq_positions = np.array([k.position for k in self.residues])
        return h_scalar, h_vector, seq_positions
    
    # getting information ---------------------------
    def get_mean_fitness(self):
        # get fitness
        if len(self.fitness) == 0:
            return None
        fitness = self.fitness
        # apply penalties
        if len(self.penalties) > 0:
            try:
                fitness = [k * (1 -self.penalties[n]) for n, k in enumerate(self.fitness)]
            except IndexError:
                fitness = [k * (1 - self.penalties[0]) for n, k in enumerate(self.fitness)]
        return sum(fitness) / len(fitness)
    
    def get_mean_penalty(self):
        if len(self.penalties) == 0:
            return None
        return sum(self.penalties) / len(self.penalties)
        
    def get_iterations(self):
        return len(self.fitness)
    
    def get_faces(self, phi=180, three_dim=True):
        """
        Returns the requires face or both faces. It splits the sequence in the angle phi 
        respect to hydrobobic vector (phi/2 on each side of the vector).
        returns lists of indexes and positions
        phi is called slice angle in instructor!!
        """
        increment = 1  # only used if three_dim is True
        # transform into radians
        phi = phi * np.pi / 180
        # border in terms of cosine of phi/2
        border = np.cos(phi/2)

        # set hydrophobic vector as reference
        h_moment = self.hydrophobic_vector / np.linalg.norm(self.hydrophobic_vector)  # normalization
        new_positions = []
        
        for n, pos in enumerate(self.raw_positions):
            # ensure normal vectors
            normal_pos = pos / np.linalg.norm(pos)
            new_cos = normal_pos[0]*h_moment[0]+normal_pos[1]*h_moment[1]  # dot
            new_sin = normal_pos[0]*h_moment[1]-normal_pos[1]*h_moment[0]  # cross
            if three_dim:
                new_positions.append([new_cos, new_sin, n*increment])
            else:
                new_positions.append([new_cos, new_sin])
        new_positions = np.array(new_positions)

        # split residues according to their cosine value
        # borders are on phi/2 (left and right)
        positive_face = []  # hydrophobic
        negative_face = []  # hydrophilic
        for num, pos in enumerate(new_positions):
            if pos[0] >= border:
                positive_face.append(num)
            else:
                negative_face.append(num)
        
        return positive_face, negative_face, new_positions
    
    def get_charged_res(self, charge='positive', letters=False):
        """
        Returns charged residues:
        positive, negative or both
        Return indexes by default.
        """
        if charge == 'positive':
            tester = lambda x: x > 0
        elif charge == 'negative':
            tester = lambda x: x < 0
        elif charge == 'both':
            tester = lambda x: x != 0
        else:
            logger.error('Sequence: get_charged_res: charge value not recognized')
            exit(3)
        charged = []
        for n, aa in enumerate(self.sequence):
            if tester(self.aa_charges[aa]):
                charged.append(n)
        if letters:
            charged = [self.sequence[k] for k in charged]
        return charged
    
    # checker --------------------------------------
    def check_elite(self) -> bool:
        # Checks if this sequences must be elite or not
        if self.is_top:
            logger.info(f'Sequence: {self.sequence} is elite')
            self.times_elite += 1
            self.is_elite = True
            return True
        self.is_elite = False
        return False
    
    def check_reinsertion(self, iterations_preferent=3) -> None:
        if self.is_discarded:
            self.reinsertions += 1
            logger.info(f'Sequence: {self.sequence} was reinserted ({self.reinsertions})')
            self.is_discarded = False
        if self.reinsertions >= iterations_preferent:
            self.is_preferent = True
            logger.info(f'Sequence: {self.sequence} is preferent')
    
    def check_resurrection(self):
        self.is_resurrected = True
        self.resurrections += 1
        logger.info(f'Sequence: {self.sequence} is resurrected')

if __name__ == '__main__':
    pass

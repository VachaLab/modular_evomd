# === sequence.py ===
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Sequence:
    name = 'sequence'
    aa_charges = {
        'A': 0.0, 'N': 0.0, 'C': 0.0, 'Q': 0.0,
        'G': 0.0, 'H': 0.0, 'I': 0.0, 'L': 0.0,
        'M': 0.0, 'F': 0.0, 'P': 0.0, 'S': 0.0,
        'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0,
        'R': 1.0, 'K': 1.0,
        'D': -1.0, 'E': -1.0,
    }

    def __init__(self, seq, generation=0, h_scale='eisenberg') -> None:
        self.sequence = seq
        self.generation = generation  # in which generation it was created
        self.fitness = []
        self.penalties = []
        # --- properties ---
        self.hydrophobic_scale = h_scale
        self.hydrophobic_moment, self.hydrophobic_vector, self.raw_positions = self.compute_hydrophobic_moment()
        self.charge = self.compute_charge()
        self.n_ter_charge = self.aa_charges[self.sequence[0]]
        self.c_ter_charge = self.aa_charges[self.sequence[-1]]
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
        charge = 0
        for letter in self.sequence:
            charge += self.aa_charges[letter]
        return charge

    def compute_hydrophobic_moment(self, scale=None, theta=100):
        hydrophobicity_scales = {
            'eisenberg': {
                'A': 0.25, 'R': -1.76, 'N': -0.64, 'D': -0.72, 'C': 0.04, 'Q': -0.69,
                'E': -0.62, 'G': 0.16, 'H': -0.4,  'I': 0.73, 'L': 0.53, 'K': -1.1,
                'M': 0.26, 'F': 0.61, 'P': -0.07, 'S': -0.26, 'T': -0.18, 'W': 0.37,
                'Y': 0.02, 'V': 0.54,
            },
            'kyte-doolittle': {
                'A': 1.8,  'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,  'Q': -3.5,
                'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,  'L': 3.8,  'K': -3.9,
                'M': 1.9,  'F': 2.8,  'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
                'Y': -1.3, 'V': 4.2,
            },
            'wimley-white': {
                'A': 0.17, 'R': 0.81, 'N': 0.42, 'D': 1.23, 'C': -0.24, 'Q': 0.58,
                'E': 2.02, 'G': 0.01, 'H': 0.96, 'I': -0.31, 'L': -0.56, 'K': 0.99,
                'M': -0.23, 'F': -1.13, 'P': 0.45, 'S': 0.13, 'T': 0.14, 'W': -1.85,
                'Y': -0.94, 'V': 0.07,
            },
            'fauchere-pliska': {
                'A': 0.31,  'R': -1.01, 'N': -0.60, 'D': -0.77, 'C': 1.54,  'Q': -0.22,
                'E': -0.64, 'G': 0.00,  'H': 0.13,  'I': 1.80,  'L': 1.70,  'K': -0.99,
                'M': 1.23,  'F': 1.79,  'P': 0.72,  'S': -0.04, 'T': 0.26,  'W': 2.25,
                'Y': 0.96,  'V': 1.22,
            },
        }
        # choose scale
        if scale:
            hydrophobicity = hydrophobicity_scales[scale]
        else:
            hydrophobicity = hydrophobicity_scales[self.hydrophobic_scale]
        # compute hydrophobic vector
        seq_positions = []
        rad = np.deg2rad(theta)
        x, y = 0.0, 0.0
        # --- Vector sum for hydrophobic moment ---
        for i, aa in enumerate(self.sequence):
            h = hydrophobicity.get(aa, 0.0)
            angle = i * rad
            cos = np.cos(angle)
            sin = np.sin(angle)
            seq_positions.append([cos, sin])
            x += h * cos
            y += h * sin
        seq_positions = np.array(seq_positions)
        h_vector = np.array([x, y])
        h_scalar = np.linalg.norm(h_vector)
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
    
    def get_hface_indexes(self, positive=True, theta=100):
        """
        Returns a list of indexes and cross values with normal vector as reference
        """
        seq_positions = []
        rad = np.deg2rad(theta)
        # --- Vector sum for hydrophobic moment ---
        for i, aa in enumerate(self.sequence):
            angle = i * rad
            cos = np.cos(angle)
            sin = np.sin(angle)
            seq_positions.append([cos, sin])
        seq_positions = np.array(seq_positions)
        h_moment = self.hydrophobic_vector / np.linalg.norm(self.hydrophobic_vector)
        normal = np.array([-h_moment[1], h_moment[0]])
        # select residues by side
        pos_index = []
        pos_cross = []
        neg_index = []
        neg_cross = []
        for index, pos in enumerate(seq_positions):
            cross = normal[0] * pos[1] - normal[1] * pos[0]
            if cross > 0:
                pos_cross.append(cross)
                pos_index.append(index)
            else:
                neg_cross.append(cross)
                neg_index.append(index)
        if positive:
            return pos_index ,pos_cross
        else:
            return neg_index, neg_cross
        
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

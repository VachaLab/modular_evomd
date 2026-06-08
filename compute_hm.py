# === compute_hm.py ===
import logging
logger = logging.getLogger(__name__)


def constructor_method(sequence) -> None:
    print(f'Nothing to create for {sequence}')

def calculator_method(sequence) -> None:
    print('Check sequence')
    print(sequence)

def calculator_check(sequence) -> bool:
    try:
        value = sequence.hydrophobic_index
        print(f'Sequence is ready: {value}')
    except:
        print('Where is the sequence?')
        return False
    try:
        times = sequence.check_times
        if times > 0:
            return True
    except:
        sequence.check_times = 1
        return False
    return False

def analyzer_method(sequence) -> float:
    from sequence_geometry import compute_hm_scalar, compute_helix_positions
    positions = compute_helix_positions(sequence)
    hm = compute_hm_scalar(sequence, positions)
    with open('fitness.txt', 'w') as f:
        f.write(f'seq: {sequence}\nhm: {hm}\ncharge: {sequence.charge}\n')
    return hm

if __name__ == '__main__':
    pass


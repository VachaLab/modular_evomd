# === test.py ===
from radial_sequence import get_radial
import sys

def read_csv(csv_file):
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    lines = [[k.split(',')[0], k.split(',')[1]] for k in lines[1:]]
    return lines

def main():
    lines = read_csv(sys.argv[1])
    print(lines)
    radials = []
    for seq, fit in lines:
        radial = get_radial(seq)
        radials.append([radial, fit])
    print(radials)
    with open('sequences_radial.csv', 'w') as f:
        f.write('Sequence,Fitness\n')
        for seq, fit in radials:
            text = f'{seq},{fit}'
            f.write(text)
    print('done')

if __name__ == '__main__':
    main()

    

import numpy as np
import sys

def main():
    # input file is always first argument
    infile = sys.argv[1]
    # infile = 'pull_pullx.xvg'
    # number of points is the second argument
    num_points = int(sys.argv[2])
    # num_points = 15

    # max value
    try:
        max_pos = float(sys.argv[3])
    except IndexError:
        max_pos = None

    lines = []
    with open(infile, 'r') as fi:
        lines = fi.readlines()
    clean = [k for k in lines if k[0] not in ['#', '@']]
    points = [[n, float(k.split()[1])] for n, k in enumerate(clean)]
    points.sort(key=lambda x: x[1])

    # set max value
    if max_pos:
        points = [k for k in points if k[1] <= max_pos]

    equidistant = np.linspace(points[0][1], points[-1][1], num_points)
    indexes = np.searchsorted([k[1] for k in points], equidistant)
    confs = [points[k][0] for k in indexes]
    # print selected points
    for p in confs:
        print(p, end=' ')
    print()


if __name__ == '__main__':
    main()


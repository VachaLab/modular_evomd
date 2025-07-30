# === free_energy.py ===
# logging instead of print
import logging
logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import os
from contextlib import contextmanager

class DataSet:
    def __init__(self, filename):
        self.name = filename
        self.xlabel = ''
        self.ylabel = ''

    def read_xvg(self, xvgfile, column=1):
        with open(xvgfile, 'r') as f:
            lines = f.readlines()
        # reading instructions and data
        xaxis = ''
        yaxis = ''
        xdata = []
        ydata = []
        for line in lines:
            if line[0] == '#':
                continue
            if line[0] == '@':
                if 'xaxis' in line:
                    xaxis = line.split('"')[1]
                if 'yaxis' in line:
                    yaxis = line.split('"')[1]
                continue
            xdata.append(float(line.split()[0]))
            ydata.append(float(line.split()[column]))
        return np.array(xdata), xaxis, np.array(ydata), yaxis

class Profile(DataSet):
    def __init__(self, filename, max=None, min_range=None):
        super().__init__(filename)
        self.xdata = []
        self.ydata = []
        self.xaxis = ''
        self.yaxis = ''
        self.yerror = []
        self.have_error = False
        self.max_pos = max
        self.min_range = min_range
        self.min = 0
        self.max = 0
        self.dg = 0
        self.error = 0
        self.indice_max = 0
        self.indice_min = 0
        self.indice_min_range = 0

    def get_profile(self):
        self.xdata, self.xaxis, self.ydata, self.yaxis = self.read_xvg(self.name, column=1)

        # Find minimum
        if self.min_range:
            # find closest index to self.min_range value
            test_dif = 1000
            for i, n in enumerate(self.xdata):
                if abs(n - self.min_range) > test_dif:
                    continue
                self.indice_min_range = i
                test_dif = abs(n - self.min_range)
            self.indice_min = np.argmin(self.ydata[:self.indice_min_range])
            self.min = self.ydata[self.indice_min]
            print(self.xdata[self.indice_min])
        else:
            self.indice_min = np.argmin(self.ydata)
            self.min = self.ydata[self.indice_min]

        # Find the closest value to max
        if self.max_pos:
            test_dif = 1000
            for i, n in enumerate(self.xdata):
                if abs(n - self.max_pos) > test_dif:
                    continue
                self.indice_max = i
                test_dif = abs(n - self.max_pos)
            self.max = self.ydata[self.indice_max]
        else:
            self.indice_max = np.argmax(self.ydata)
            self.max = self.ydata[self.indice_max]
#        print(f'**** {self.indice_max} ****')
#        print(f'**** {self.max} ****')
        self.dg = round(self.max - self.min, 2)

    def get_errors(self, ferrors):
        _, _, self.yerror, _ = self.read_xvg(ferrors, column=2)
        # compute error in DG
        self.error = (self.yerror[self.indice_max] ** 2) + (self.yerror[self.indice_min] ** 2)
        self.error = round(np.sqrt(self.error), 2)
        self.have_error = True

class Plot:
    def __init__(self, profile1, profile2, output='out.png', show=True, save=True, text=True, x_offset=1.5, y_offset=20):
        self.fig, self.ax1 = plt.subplots(1, 1)
        self.profile1 = profile1
        self.profile2 = profile2
        self.output = output
        self.show = eval(show)
        self.save = eval(save)
        self.text = eval(text)
        self.x_offset = x_offset  # Ajuste para la posición en X del texto
        self.y_offset = y_offset  # Ajuste para la posición en Y del texto

    def plot_profile(self):
        # Plotting the first profile
        self.ax1.plot(self.profile1.xdata, self.profile1.ydata, lw=1.5, c='blue', label='Profile 1')
        
        # Plotting the second profile
        self.ax1.plot(self.profile2.xdata, self.profile2.ydata, lw=1.5, c='red', label='Profile 2')

        # Marking the maxima and minima for profile 1
        self.ax1.plot(self.profile1.xdata[self.profile1.indice_max], self.profile1.max, 'bo')  # Max for profile 1
        self.ax1.plot(self.profile1.xdata[self.profile1.indice_min], self.profile1.min, 'go')  # Min for profile 1
        self.ax1.plot([self.profile1.xdata[self.profile1.indice_max], self.profile1.xdata[self.profile1.indice_max]], [0, self.profile1.max], 'b--', lw=0.8)  # Line for max
        self.ax1.plot([self.profile1.xdata[self.profile1.indice_min], self.profile1.xdata[self.profile1.indice_min]], [0, self.profile1.min], 'g--', lw=0.8)  # Line for min

        # Marking the maxima and minima for profile 2
        self.ax1.plot(self.profile2.xdata[self.profile2.indice_max], self.profile2.max, 'ro')  # Max for profile 2
        self.ax1.plot(self.profile2.xdata[self.profile2.indice_min], self.profile2.min, 'yo')  # Min for profile 2
        self.ax1.plot([self.profile2.xdata[self.profile2.indice_max], self.profile2.xdata[self.profile2.indice_max]], [0, self.profile2.max], 'r--', lw=0.8)  # Line for max
        self.ax1.plot([self.profile2.xdata[self.profile2.indice_min], self.profile2.xdata[self.profile2.indice_min]], [0, self.profile2.min], 'y--', lw=0.8)  # Line for min

        # Calculating and plotting the difference in DG
        self.delta_dg = round(self.profile1.dg - self.profile2.dg, 2)
        self.ax1.set_ylabel('PMF (kJ/mol)')
        self.ax1.set_xlabel('Reaction coordinate (nm)')
        
        if self.text:
            # set position
            text_x = self.profile1.xdata[self.profile1.indice_max] - self.x_offset
            text_y = self.profile1.ydata[self.profile1.indice_min] + self.y_offset
            # create text
            txt = r'$\Delta G_1 = $' + f'{self.profile1.dg}' + ' kJ/mol'
            txt += '\n' + r'$\Delta G_2 = $' + f'{self.profile2.dg}' + ' kJ/mol'
            txt += '\n' + r'$\Delta \Delta G = $' + f'{self.delta_dg}' + ' kJ/mol'
            # plot text
            self.ax1.text(text_x, text_y, txt, fontsize=10)
        self.ax1.set_xlim(self.profile1.xdata[0], self.profile1.xdata[-1])

        self.ax1.legend()

    def ending(self):
        if self.save:
            plt.savefig(self.output, dpi=300)
        if self.show:
            plt.show()

def parser():
    parser = argparse.ArgumentParser(description ='Plot PMF profile')
    parser.add_argument('--profile1', help='XVG profile to plot',  type=str, default=None)
    parser.add_argument('--profile2',   help='XVG histogram to plot',type=str, default=None)
    parser.add_argument('-save', '--save',  help='True: save plot, False: not save', default='True')
    parser.add_argument('-text', '--text',  help='True: include text, False: not include text', default='True')
    parser.add_argument('-out', '--out',  help='Output name', type=str, default='output.png')
    parser.add_argument('-show', '--show',  help='True: show, False: not show', default='True')
    parser.add_argument('-max', '--max', help=r'Maximum value to compute $\Delta$G', type=float, default=None)
    parser.add_argument('-min', '--min-range', help='Look for minimum en range (-inf, --min-range)', type=float, default=None)

    args = parser.parse_args()
    # verify if argument is None
    if args.profile1 is None:
        parser.print_help()  # print help
        sys.exit(1)  # exit with error code 1

    return args


def main():
    prs = parser()
    profile1 = Profile(prs.profile1, prs.max, prs.min_range)
    profile1.get_profile()
    profile2 = Profile(prs.profile2, prs.max, prs.min_range)
    profile2.get_profile()
    # Create plot object
    plot = Plot(profile1, profile2, prs.out, prs.show, prs.save, prs.text)
    plot.plot_profile()
    plot.ending()

@contextmanager
def change_dir(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

def analyzer_method(sequence) -> float:
    """
    receives a sequence and compute fitness function
    returns the fitness function as float
    """
    # list of membranes
    membranes = ['ecoli', 'human']
    file1 = membranes[0] + '/result.txt'
    file2 = membranes[1] + '/result.txt'
    results = []
    with open(file1, 'r') as fr:
        lines = fr.readlines()
        results.append(int(lines[0].strip()))
    return sum(results) / len(results)


if __name__ == '__main__':
    main()


# === analyzer_umbrella.py ===
# logging instead of print
import logging
logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from scipy.optimize import curve_fit

import sys
import argparse
import os
from contextlib import contextmanager

import subprocess


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
    def __init__(self, filename, name='profile', max=None):
        super().__init__(filename)
        self.profile_name = name
        self.xdata = []
        self.ydata = []
        self.xaxis = ''
        self.yaxis = ''
        self.yerror = []
        self.have_error = False
        self.max_pos = max
        self.min = 0
        self.max = 0
        self.dg = 0
        self.error = 0
        self.indice_max = 0
        self.indice_min = 0

    def get_profile(self):
        self.xdata, self.xaxis, self.ydata, self.yaxis = self.read_xvg(self.name, column=1)
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
        self.normal_minimum()
    
    def get_cubicspline_profile_back(self):
        self.xdata, self.xaxis, self.ydata, self.yaxis = self.read_xvg(self.name, column=1)
        # parameters used in removing holes
        # These will be included as argument options in the future
        min_depth = 10  # minimum depth to be considered as hole (kJ/mol)
        window = 3  # number of points to compute depths
        exclude_window = 5  # number of points to remove for each minimum
        min_position = 2.3  # remove all the minima from this point
        # peaks
        inv_y = -self.ydata
        minima_indices, _ = find_peaks(inv_y)
        # compute depths
        # Guardar resultados
        depths = []
        positions = []
        for idx in minima_indices:
            y_min = self.ydata[idx]

            # Izquierda
            left_region = self.ydata[idx - window:idx] if idx - window >= 0 else self.ydata[0:idx]

            # Derecha
            right_region = self.ydata[idx + 1:idx + window + 1]

            # Máximo local a izquierda y derecha (si existen)
            left_max = np.max(left_region) if len(left_region) > 0 else -np.inf
            right_max = np.max(right_region) if len(right_region) > 0 else -np.inf

            # Tomar el máximo menor entre ambos lados
            ref_max = min(left_max, right_max)

            # Calcular profundidad
            depth = ref_max - y_min
            
            # selecionar solo mínimos con profundidad adecuada
            if depth >= min_depth:
                depths.append(depth)
                positions.append(idx)
            
            # Remover los mínimos adicionales
            new_x = []
            new_y = []
            excl_list = []
            for index in positions:
                if self.xdata[index] < min_position:
                    continue
                excl_list.extend(range(index - exclude_window, index + exclude_window))
            for index, (x_pos, y_pos) in enumerate(zip(self.xdata, self.ydata)):
                if index in excl_list:
                    continue
                new_x.append(x_pos)
                new_y.append(y_pos)
            new_x = np.array(new_x)
            new_y = np.array(new_y)

            # Ajustar curva
            spline = CubicSpline(new_x, new_y)
            self.ydata = spline(self.xdata)

            # calcular mínimo y máximo
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
    
    def get_cubicspline_profile(self):
        self.xdata, self.xaxis, self.ydata, self.yaxis = self.read_xvg(self.name, column=1)
        # Remove artifacts
        min_position = 2.5  # remove all the minima from this point
        # where is min_position?
        min_pos_index = np.argmin(np.array([abs(k-min_position) for k in self.xdata]))
        # remove zeros after min_pos_index
        new_ydata = []
        new_xdata = []
        for x,y in zip(self.xdata[min_pos_index:], self.ydata[min_pos_index:]):
            if not y > 15:
                continue
            new_xdata.append(x)
            new_ydata.append(y)
        self.xdata = np.concatenate((self.xdata[:min_pos_index], np.array(new_xdata)))
        self.ydata = np.concatenate((self.ydata[:min_pos_index], np.array(new_ydata)))

        # Ajustar polinomio de 5to orden
        coeficientes = np.polyfit(self.xdata, self.ydata, 10)  # Devuelve [a5, a4, ..., a0]
        polinomio = np.poly1d(coeficientes)

        # Evaluar el ajuste
        x_fit = np.linspace(min(self.xdata), max(self.xdata), 100)
        y_fit = polinomio(x_fit)
        self.xdata = x_fit
        self.ydata = y_fit
        
        # calcular mínimo y máximo
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
        self.normal_minimum()

    def get_errors(self, ferrors):
        _, _, self.yerror, _ = self.read_xvg(ferrors, column=2)
        # compute error in DG
        self.error = (self.yerror[self.indice_max] ** 2) + (self.yerror[self.indice_min] ** 2)
        self.error = round(np.sqrt(self.error), 2)
        self.have_error = True

    def normal_minimum(self):
        self.ydata = self.ydata - self.min
        self.max = self.max - self.min
        self.min = 0.0

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
        self.ax1.plot(self.profile1.xdata, self.profile1.ydata, lw=1.5, c='blue', label=self.profile1.profile_name)
        
        # Plotting the second profile
        self.ax1.plot(self.profile2.xdata, self.profile2.ydata, lw=1.5, c='red', label=self.profile2.profile_name)

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
            txt = rf'$\Delta G_{{{self.profile1.profile_name}}} = $' + f'{self.profile1.dg}' + ' kJ/mol'
            txt += '\n' + rf'$\Delta G_{{{self.profile2.profile_name}}} = $' + f'{self.profile2.dg}' + ' kJ/mol'
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
    
    def closefig(self):
        plt.close(self.fig)

def parser():
    parser = argparse.ArgumentParser(description ='Plot PMF profile')
    parser.add_argument('--profile1', help='XVG profile to plot',  type=str, default=None)
    parser.add_argument('--profile2',   help='XVG histogram to plot',type=str, default=None)
    parser.add_argument('--name1', help='Name for profile 1', type=str, default='Profile1')
    parser.add_argument('--name2', help='Name for profile 2', type=str, default='Profile2')
    parser.add_argument('-save', '--save',  help='True: save plot, False: not save', default='True')
    parser.add_argument('-text', '--text',  help='True: include text, False: not include text', default='True')
    parser.add_argument('-out', '--out',  help='Output name', type=str, default='output.png')
    parser.add_argument('-show', '--show',  help='True: show, False: not show', default='True')
    parser.add_argument('-max', '--max', help=r'Maximum value to compute $\Delta$G', type=float, default=None)
    parser.add_argument('-spl', '--spline', help=r'Compute PMF curve with cubic spine', type=str, default='False')

    args = parser.parse_args()
    # verify if argument is None
    if args.profile1 is None:
        parser.print_help()  # print help
        sys.exit(1)  # exit with error code 1

    return args


def main():
    prs = parser()
    profile1 = Profile(prs.profile1, prs.name1, prs.max)
    profile2 = Profile(prs.profile2, prs.name2, prs.max)
    if eval(prs.spline):
        print("cubic spline")
        profile1.get_cubicspline_profile()
        profile2.get_cubicspline_profile()
    else:
        print("no cubic spline")
        profile1.get_profile()
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

def gmx_wham(membranes=['ecoli', 'human']):
    for m in membranes:
        with change_dir(m):
            file_confs = 'configurations.txt'
            nums = []
            with open(file_confs, 'r') as num_list:
                nums = num_list.readlines()[0].strip().split()
            tpr_files = open('tpr-files.dat', 'w')
            pullf_files = open('pullf-files.dat', 'w')
            for n in nums:
                tpr_files.write(f'conf{n}/umbrella.tpr\n')
                pullf_files.write(f'conf{n}/umbrella_pullf.xvg\n')
            tpr_files.close()
            pullf_files.close()
            command = [
                'gmx', 'wham',
                '-it', 'tpr-files.dat',
                '-if', 'pullf-files.dat'
            ]
            with open('gmx_wham.log', 'w') as logfile:
                proceso = subprocess.run(command, stdout=logfile, stderr=subprocess.STDOUT, text=True)

def analyzer_method(sequence) -> float:
    """
    receives a sequence and compute fitness function
    returns the fitness function as float.
    It always uses cubicsplines method.
    """
    # list of membranes
    membranes = ['ecoli', 'human']
    maximum = 5.0

    # run wham
    gmx_wham(membranes=membranes)

    # change profile names
    profile1_file = os.path.join(membranes[0], 'profile.xvg')
    profile2_file = os.path.join(membranes[1], 'profile.xvg')
    # verify files
    if not os.path.exists(profile1_file) or not os.path.exists(profile2_file):
        logger.error("One or both profile files are missing.")
        return np.nan
    # if files exist, analyze them
    logger.info(f'Analyzing sequence {str(sequence)}')
    logger.info(f'Directory: {sequence.last_iter_dir}')
    profile1 = Profile(profile1_file, membranes[0], maximum)
    profile1.get_cubicspline_profile()
    profile2 = Profile(profile2_file, membranes[1], maximum)
    profile2.get_cubicspline_profile()
    # Create plot object
    plot = Plot(profile1, profile2, output='fitness_value.png', show='False', save='True', text='True')
    plot.plot_profile()
    plot.ending()
    plot.closefig()
    return plot.delta_dg


if __name__ == '__main__':
    main()


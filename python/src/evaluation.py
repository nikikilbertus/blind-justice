"""
This module evaluates the results from multiple runs and generates plots.
"""

import csv
import os
from itertools import product
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cycler

from data import read_compas


class Evaluation(object):

    # Convert abbreviations to full words for plot labels
    optimizer_labels = {'lagrange': 'Lagr.',
                        'iplb': 'iplb',
                        'projected': 'proj.',
                        'baseline': 'bl'}
    dataset_labels = {'adult': 'Adult',
                      'bank': 'Bank',
                      'synthetic': 'Synthetic',
                      'compas': 'COMPAS',
                      'sqf': 'SQF',
                      'german': 'German'}

    def __init__(self, filepath, fontsize=12):
        """
        Initialize an evaluation instance.

        Args:
            filepath: The csv file containing the results
            fontsize: Fontsize used in plot titles and labels
        """
        self.filepath = filepath
        self.fontsize = fontsize
        self.r = pd.read_csv(filepath)
        # We will rely on the values being sorted by the constraints
        self.r.sort_values(by='constraint', inplace=True)
        # Reset the indices after the sorting
        self.r.reset_index(drop=True, inplace=True)

        # Get the parameters for which results are present
        self.params = {
            'datasets': get_distinct_values(self.r, 'dataset'),
            'optimizers': get_distinct_values(self.r, 'optimizer'),
            'fairness': get_distinct_values(self.r, 'fairness'),
            'approximations': get_distinct_values(self.r, 'approximation'),
            'constraints': get_distinct_values(self.r, 'constraint'),
            'batchsizes': get_distinct_values(self.r, 'batchsize'),
            'bits': list(set([tuple(a)
                              for a in self.r[['nbits', 'nintbits']].values])),
            'max_sample_size': get_distinct_values(self.r, 'max_sample_size'),
        }

    def print_value_summary(self):
        """Print the parameters for which results are present."""
        pprint(self.params)

    def plot_accuracy(self,
                      datasets=None,
                      fairness='ppercent',
                      batchsize=64,
                      nbits=0,
                      nintbits=0,
                      show=['valid_accuracy'],
                      save_fig=False,
                      ppercent_on_x=True,
                      per_plot_size=3):
        """Compare accuracies from optimization techniques across datasets."""
        if datasets is None:
            datasets = self.params['datasets']
        n_data = len(datasets)
        optimizers = self.params['optimizers']
        optimizers.remove('baseline')
        n_optimizers = len(optimizers)
        n_approx = len(self.params['approximations'])
        fig, axs = plt.subplots(1, n_data,
                                figsize=(per_plot_size * n_data, per_plot_size),
                                sharex=(not ppercent_on_x))

        # Number of colors for lines is number of optimizers
        # Get color cycle list with linestyles for two different approximations
        cls = plt.rcParams['axes.prop_cycle'].by_key()['color'][:n_optimizers]
        if n_approx == 1:
            linestyle = {self.params['approximations'][0]: '-'}
        elif n_approx == 2:
            cls = [cl for cls in list(zip(cls, cls)) for cl in cls]
            linestyle = {'none': '-', 'secureml': '--'}
        else:
            raise ValueError('Only support 1 or 2 different approximations.')

        # Loop over datasets for different plots
        for j, data in enumerate(datasets):
            if n_data == 1:
                curax = axs
            else:
                curax = axs[j]
            lines = []
            leglabels = []
            select = self.r.loc[
                (self.r['dataset'] == data) &
                (self.r['fairness'] == fairness) &
                (self.r['optimizer'] == 'baseline')]
            accs = []
            constraints = get_distinct_values(select, 'constraint')
            for c in constraints:
                acc = select.loc[select['constraint'] == c, show].values
                if acc.size > 0:
                    accs.append(acc[0])
                else:
                    accs.append(0 if len(show) == 1 else [0]*len(show))
            if ppercent_on_x:
                x_vals = self.get_ppercent(select)
            else:
                x_vals = constraints
            if ppercent_on_x:
                td, = curax.plot(x_vals, accs, 'r', label='baseline')
            else:
                td, = curax.semilogx(x_vals, accs, 'r', label='baseline')
            lines.append(td)
            leglabels.append(self.optimizer_labels['baseline'])
            # Loop over optimizers for different colors
            curax.set_prop_cycle(cycler('color', cls))
            for k, opt in enumerate(optimizers):
                # Loop over approximations for different line styles
                for approx in self.params['approximations']:
                    select = self.r.loc[
                        (self.r['approximation'] == approx) &
                        (self.r['dataset'] == data) &
                        (self.r['fairness'] == fairness) &
                        (self.r['batchsize'] == batchsize) &
                        (self.r['optimizer'] == opt) &
                        (self.r['nbits'] == nbits) &
                        (self.r['nintbits'] == nintbits)]
                    accs = []
                    constraints = get_distinct_values(select, 'constraint')
                    for c in constraints:
                        acc = select.loc[select['constraint'] == c, show].values
                        if acc.size > 0:
                            accs.append(acc[0])
                        else:
                            accs.append(0 if len(show) == 1 else [0]*len(show))
                    if ppercent_on_x:
                        x_vals = self.get_ppercent(select)
                    else:
                        x_vals = constraints
                    if nbits != 0:
                        bitlabel = (str(int(nbits)) + ' bits')
                    else:
                        bitlabel = 'float'
                    pltlabel = opt + ', ' + approx + ', ' + bitlabel
                    if ppercent_on_x:
                        td, = curax.plot(x_vals, accs, label=pltlabel,
                                         linestyle=linestyle[approx])
                    else:
                        td, = curax.semilogx(x_vals, accs, label=pltlabel,
                                             linestyle=linestyle[approx])
                    lines.append(td)
                    leglabels.append(self.optimizer_labels[opt] + ', ' + approx)
            try:
                baseline_acc = select['baseline_accuracy'].values[0]
                curax.axhline(baseline_acc, color='k', linestyle='--',
                              label='baseline')
            except KeyError:
                pass
            curax.grid(axis='y')
            curax.set_title(self.dataset_labels[data])
        if n_data == 1:
            axs.set_ylabel('constraint', fontsize=self.fontsize)
        else:
            axs[n_data-1].set_ylabel('accuracy', fontsize=self.fontsize)
        fig.legend(lines, leglabels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        fig.tight_layout()
        fig.show()
        if save_fig:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-$d_%H-%M-%S")
            figname = "accuracy_comparison_" + timestamp + ".pdf"
            fig.savefig(figname,  bbox_inches='tight')

    def plot_frac_in_positive(self,
                              datasets=None,
                              compas_index=1,
                              batchsize=64,
                              fairness='ppercent',
                              optimizer='lagrange',
                              save_fig=False,
                              ppercent_on_x=True,
                              per_plot_size=3):

        if datasets is None:
            datasets = self.params['datasets']
        n_data = len(datasets)
        n_approximations = len(self.params['approximations'])
        n_bits = len(self.params['bits'])

        approximations = self.params['approximations']
        bits = self.params['bits']
        show = ('valid_z0_in_y1_test', 'valid_z1_in_y1_test')

        # Draw lines for the positive fractions in both demographic groups
        labels = ['z=0', 'z=1']
        linestyle = ['-', '--']
        n_lines = n_approximations * n_bits
        cls = plt.rcParams['axes.prop_cycle'].by_key()['color'][:n_lines+1]
        cls = [cl for cls in list(zip(cls, cls)) for cl in cls]

        # We need to load the compas dataset again to compute fractions
        if 'compas' in datasets:
            _, Xte, _, yte, _, Zte = read_compas()

        fig, axs = plt.subplots(1, n_data,
                                figsize=(per_plot_size * n_data, per_plot_size),
                                sharex=(not ppercent_on_x))

        # Loop through datasets for different plots
        for j, data in enumerate(datasets):
            if n_data == 1:
                curax = axs
            else:
                curax = axs[j]
            curax.set_prop_cycle(cycler('color', cls))
            lines = []
            leglabels = []

            select = self.r.loc[
                (self.r['dataset'] == data) &
                (self.r['fairness'] == fairness) &
                (self.r['optimizer'] == 'baseline')]
            accs = []
            constraints = get_distinct_values(select, 'constraint')
            if data == 'compas':
                accs = self.get_frac_compas(select, compas_index,
                                            Xte, Zte)
            else:
                for c in constraints:
                    acc = select.loc[select['constraint'] == c, show].values
                    if acc.size > 0:
                        accs.append(acc[0])
                    else:
                        accs.append([0] * len(show))
            accs = np.array(accs)
            for acc, label, ls in zip(accs.T, labels, linestyle):
                pltlabel = 'baseline ' + label
                if ppercent_on_x:
                    x_vals = self.get_ppercent(select)
                    td, = curax.plot(x_vals, acc, label=pltlabel, linestyle=ls)
                else:
                    x_vals = constraints
                    td, = curax.semilogx(x_vals, acc, label=pltlabel,
                                         linestyle=ls)
                lines.append(td)
                leglabels.append('baseline')

            for approx, (nbits, nintbits) in list(product(approximations, bits)):
                select = self.r.loc[
                    (self.r['approximation'] == approx) &
                    (self.r['dataset'] == data) &
                    (self.r['fairness'] == fairness) &
                    (self.r['batchsize'] == batchsize) &
                    (self.r['optimizer'] == optimizer) &
                    (self.r['nbits'] == nbits) &
                    (self.r['nintbits'] == nintbits)]
                accs = []
                constraints = get_distinct_values(select, 'constraint')
                if data == 'compas':
                    accs = self.get_frac_compas(select, compas_index,
                                                Xte, Zte)
                else:
                    for c in constraints:
                        acc = select.loc[select['constraint'] == c, show].values
                        if acc.size > 0:
                            accs.append(acc[0])
                        else:
                            accs.append([0] * len(show))
                if ppercent_on_x:
                    x_vals = self.get_ppercent(select)
                else:
                    x_vals = constraints
                accs = np.array(accs)
                for acc, label, ls in zip(accs.T, labels, linestyle):
                    if nbits != 0:
                        bitlabel = (str(int(nbits)) + ' bits')
                    else:
                        bitlabel = 'float'
                    pltlabel = approx + ', ' + bitlabel + ', ' + label
                    if ppercent_on_x:
                        td, = curax.plot(x_vals, acc, label=pltlabel,
                                         linestyle=ls)
                    else:
                        td, = curax.semilogx(x_vals, acc, label=pltlabel,
                                             linestyle=ls)
                    lines.append(td)
                    leglabels.append(pltlabel)
            curax.grid(axis='y')
            curax.set_title(self.dataset_labels[data], fontsize=self.fontsize)
            curax.set_xlabel('constraint', fontsize=self.fontsize)
        if n_data == 1:
            axs.set_xlabel('constraint', fontsize=self.fontsize)
        else:
            pass
        fig.legend(lines, leglabels, loc='lower center',
                   bbox_to_anchor=(0.5, -0.1), ncol=2)
        fig.tight_layout()
        fig.show()
        if save_fig:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-$d_%H-%M-%S")
            figname = "ppercent_fracs_" + timestamp + ".pdf"
            fig.savefig(figname,  bbox_inches='tight')

    def get_ppercent(self, select):
        """For a number of constraitns get the corresponding p percent values"""
        y1z0 = select['valid_z0_in_y1_test'].values
        y1z1 = select['valid_z1_in_y1_test'].values
        frac1 = np.divide(y1z0, y1z1, out=np.full_like(y1z0, np.inf),
                          where=(y1z1 != 0))
        frac2 = np.divide(y1z1, y1z0, out=np.full_like(y1z1, np.inf),
                          where=(y1z0 != 0))
        return np.minimum(frac1, frac2) * 100.

    def get_frac_compas(self, select, index, Xte, Zte):
        accs = []
        for c in select['constraint']:
            weight = select.loc[select['constraint'] == c, 'valid_weights'].values
            try:
                failed = False
                if weight.size > 0:
                    weight = weight[0]
                else:
                    failed = True
                try:
                    isnan = np.isnan(weight)
                except:
                    isnan = False
                if not isnan:
                    if not isinstance(weight, np.ndarray):
                        weight_str = weight.replace('\n', '')\
                            .replace('[', '')\
                            .replace(']', '')\
                            .split()
                        weight = [float(i) for i in weight_str]
                    weight = np.array(weight, dtype=float)
                    I0 = (Zte[:, index].ravel() <= 0)
                    I1 = (Zte[:, index].ravel() > 0)
                    X0, X1 = Xte[I0, :], Xte[I1, :]
                    n0, n1 = X0.shape[0], X1.shape[0]
                    z0iny1 = np.sum(np.round(X0 @ weight) == 1) / n0
                    z1iny1 = np.sum(np.round(X1 @ weight) == 1) / n0
                else:
                    failed = True
            except:
                failed = True
            if not failed:
                accs.append([z0iny1, z1iny1])
            else:
                accs.append([np.nan, np.nan])
        return accs

    def save_csv(self, name, table):
        with open(name, 'w') as f:
            fw = csv.writer(f, delimiter=',')
            for row in table:
                fw.writerow(row)

    def export_tables(self,
                      basedir='../doc/tables/ppercent/',
                      fairness='ppercent',
                      compas_index=1,
                      batchsize=64):
        _, Xte, _, yte, _, Zte = read_compas()

        writeout = ['constraint',
                    'valid_accuracy',
                    'valid_train_accuracy',
                    'valid_z0_in_y1_train',
                    'valid_z1_in_y1_train',
                    'valid_z0_in_y1_test',
                    'valid_z1_in_y1_test',
                    'ppercent']

        compas_writeout = ['constraint',
                           'valid_accuracy',
                           'valid_train_accuracy',
                           'valid_constraint_satisfied',
                           'valid_max_constraint_value',
                           'valid_z0_in_y1_test',
                           'valid_z1_in_y1_test',
                           'ppercent']

        if not os.path.exists(basedir):
            os.makedirs(basedir)

        # write out all other data
        settings = product(self.params['datasets'],
                           self.params['optimizers'],
                           self.params['bits'],
                           self.params['approximations'])
        for data, opt, (nbits, nintbits), approx in list(settings):
            select = self.r.loc[
                (self.r['approximation'] == approx) &
                (self.r['dataset'] == data) &
                (self.r['fairness'] == fairness) &
                (self.r['batchsize'] == batchsize) &
                (self.r['optimizer'] == opt) &
                (self.r['nbits'] == nbits) &
                (self.r['nintbits'] == nintbits)]
            name = basedir + "{}-{}-{}-{}.csv".format(data, opt, approx,
                                                      int(nbits))

            if data != 'compas':
                select['ppercent'] = self.get_ppercent(select)
                select[writeout].to_csv(name, index=False)
            else:
                ziny1 = self.get_frac_compas(select, compas_index, Xte, Zte)
                ziny1 = np.array(ziny1, dtype=float)
                new_select = select.copy(deep=True)
                new_select['valid_z0_in_y1_test'] = ziny1[:, 0]
                new_select['valid_z1_in_y1_test'] = ziny1[:, 1]
                new_select['ppercent'] = self.get_ppercent(new_select)
                new_select = new_select[compas_writeout]
                new_select.to_csv(name, index=False)

            name = basedir + "{}-baseline.csv".format(data)
            baseline_acc = select['baseline_accuracy'].values[0]
            self.save_csv(name, [["baseline"]] + [[baseline_acc]])

        # write only ppercent baseline here
        for data in self.params['datasets']:
            select = self.r.loc[
                (self.r['dataset'] == data) &
                (self.r['fairness'] == fairness) &
                (self.r['optimizer'] == 'baseline')]
            select['ppercent'] = self.get_ppercent(select)
            name = basedir + "{}-ppercent-bl.csv".format(data)
            select[writeout].to_csv(name, index=False)


def get_distinct_values(df, key):
    """Get the distinct values that are present in a given column."""
    return sorted(list(df[key].value_counts().index.values))


if __name__ == '__main__':
    evaluation = Evaluation('../results/final.csv')
    evaluation.plot_accuracy(ppercent_on_x=True)
    evaluation.plot_frac_in_positive(ppercent_on_x=True)
    # evaluation.export_tables(basedir='../tmp/')

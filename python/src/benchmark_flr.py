"""
This module is used to benchmark FairLogisticRegression.
"""

import os
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from spfpm.FixedPoint import FXfamily
from fair_logistic_regression import FairLogisticRegression
from tools import to_float


class BenchmarkFLR(object):

    def __init__(self,
                 import_path=None,
                 optimizations=('unconstrained',
                                'step-size',
                                'lagrange',
                                'iplb',
                                'projected'),
                 approximations=('none', 'linear', 'secureml'),
                 n_bits_range=(4, 8, 10, 16),
                 n_intbits_range=(4, 8, 10, 16),
                 constraints=(10., 1., 0.01),
                 update_rule='vanilla',
                 batchsize=64,
                 epochs=10,
                 t_iplb=0.5,
                 verbose=0,
                 output_type='csv',
                 output_path=None):
        """Initialize a Benchmark run for FairLogisticRegression.

        Args:
            import_path: (bool) If true, ignore other arguments and read file
            optimizations: The optimization method to use
            approximations: A list of approximations to use for the sigmoid
            n_bits_range: A list of bitnumbers to use for the fractional part
            n_intbits_range: A list of bitnumbers to use for the integral part
            constraints: A list of values for the constraint
            batchsize: The batchsize to use for all of this
            epochs: How many epochs to train in each run
            verbose: Verbosity of print outputs
            output_type: Output type (only csv implemented so far)
            output_path: The path for the output file. None is fine
        """

        if import_path is not None:
            self.import_results(import_path)
        else:
            self.optimizations = list(optimizations)
            self.approximations = list(approximations)
            self.constraints = list(constraints)
            self.n_bits_range = list(n_bits_range)
            self.n_intbits_range = list(n_intbits_range)
            bits = list(product(self.n_intbits_range, self.n_bits_range))
            bits.insert(0, (None, None))
            self.bits = bits
            self.batchsize = batchsize
            self.epochs = epochs
            self.t_iplb = t_iplb
            self.verbose = verbose
            self.output_type = output_type
            self.results = None
            self.updates = update_rule

            # Results filename
            if output_path is None or output_path == 'default':
                from datetime import datetime
                now = datetime.now().strftime("%Y%m%d-%H%M%S")
                self.output_path = os.path.join(os.curdir, now + '_results.csv')
            else:
                self.output_path = output_path

            # (How) are we storing the results?
            if self.verbose > 0:
                if self.output_type is None:
                    print("Do not store results to file.")
                elif 'csv' in self.output_type:
                    print("Store results as csv.")
                elif 'hdf' in self.output_type:
                    print("Store results as hdf.")

    def run_benchmark(self, xtr, ytr, ztr, xte, yte):
        """Run all benchmark experiments.

        Args:
            xtr: Training features (n_samples, n_features)
            ytr: Training labels (n_samples,)
            ztr: Training sensitive attributes (n_samples, n_sensitive)
            xte: Test features
            yte: Test labels

        Returns:
            None
        """
        # Quick and dirty check if internals have been set
        assert self.bits is not None, "Call `setup` first."

        results = pd.DataFrame()
        for opt in self.optimizations:
            for approx in self.approximations:
                for c in self.constraints:
                    for n_intbits, n_bits in self.bits:
                        # Choose number format and family
                        if n_intbits is None or n_bits is None:
                            family = None
                        else:
                            family = FXfamily(n_bits=n_bits,
                                              n_intbits=n_intbits)

                        run_name = "{}_{}_c={}_{}.{}".format(str(opt),
                                                             str(approx),
                                                             str(c),
                                                             str(n_intbits),
                                                             str(n_bits))
                        model = FairLogisticRegression(opt=opt,
                                                       family=family,
                                                       constraint=c,
                                                       approx=approx,
                                                       update_rule=self.updates,
                                                       t_iplb=self.t_iplb,
                                                       weight_decay=0,
                                                       momentum=0.95,
                                                       learning_rate_decay=True,
                                                       verbose=self.verbose)
                        try:
                            start = timer()
                            model.fit(xtr, ytr, ztr, Xval=xte, yval=yte,
                                      epochs=self.epochs,
                                      batchsize=self.batchsize)
                            runtime = timer() - start
                            train_acc = np.mean(ytr == model.predict(xtr))
                            test_acc = np.mean(yte == model.predict(xte))
                            failed = False
                            msg = ''
                            print("\n{} took {} seconds".format(run_name,
                                                                runtime))
                        except Exception as e:
                            runtime = None
                            train_acc = None
                            test_acc = None
                            failed = True
                            msg = repr(e)
                            print("\n{} failed".format(run_name))

                        weights = to_float(model.w)
                        constraint_values = to_float(model.constr_vals)
                        result = {'name': run_name,
                                  'optimization': opt,
                                  'constraint': c,
                                  'approximation': approx,
                                  'n_intbits': n_intbits,
                                  'n_bits': n_bits,
                                  'train_acc': train_acc,
                                  'test_acc': test_acc,
                                  'weights': weights,
                                  'constraint_values': constraint_values,
                                  'runtime': runtime,
                                  'failed': failed,
                                  'error': msg}
                        results = results.append(result, ignore_index=True)
        self.results = results
        if self.output_type is not None:
            print("Export results...", end=' ')
            self.export_results()
            print("DONE")
        print("Benchmark DONE")

    def export_results(self):
        """Write the results to a file."""
        assert self.results is not None, "No results available!"
        if self.output_type == 'csv':
            self.results.to_csv(self.output_path)
        elif self.output_type == 'hdf':
            self.results.to_hdf(self.output_path, 'results')

    def import_results(self, path):
        """Import previously saved results."""
        self.results = pd.read_csv(path, index_col=0)
        self.n_intbits_range = self._get_values('n_intbits')
        self.n_bits_range = self._get_values('n_bits')
        bits = list(product(self.n_intbits_range, self.n_bits_range))
        bits.insert(0, (None, None))
        self.bits = bits
        self.approximations = self._get_values('approximation') + [None]
        self.constraints = self._get_values('constraint')
        self.optimizations = self._get_values('optimization')

    def plot_accuracies(self):
        """Plot for comparison of all accuracies."""
        constraints = self.constraints
        approximations = self.approximations
        intbits = self.n_intbits_range
        bits = self.n_bits_range

        keys = ['train_acc', 'test_acc']
        r = self.results
        n_keys = len(keys)
        n_constr = len(constraints)

        fig = None
        for opt in self.optimizations:
            for approx in approximations:
                fig, axs = plt.subplots(n_keys, n_constr,
                                        figsize=(n_constr * 5, n_keys * 5))
                if approx is None:
                    approx_crit = r['approximation'].isnull()
                else:
                    approx_crit = r['approximation'] == approx

                for cidx, c in enumerate(constraints):
                    for ridx, key in enumerate(keys):
                        f = np.zeros((len(intbits), len(bits)))
                        bl = r.loc[(r['constraint'] == c) &
                                   (approx_crit) &
                                   (r['n_bits'].isnull()) &
                                   (r['n_intbits'].isnull()) &
                                   (r['optimization'] == opt),
                                   key].astype(float).values[0]

                        for i, ib in enumerate(intbits):
                            for j, b in enumerate(bits):
                                f[i, j] = r.loc[(r['constraint'] == c) &
                                                (approx_crit) &
                                                (r['n_bits'] == b) &
                                                (r['n_intbits'] == ib) &
                                                (r['optimization'] == opt),
                                                key].astype(float).values

                        ax = axs[ridx, cidx]
                        ax.imshow(f, vmin=0, vmax=1)
                        ax.set_xticks(range(len(bits)))
                        ax.set_yticks(range(len(intbits)))
                        ax.set_xticklabels([str(i) for i in bits])
                        ax.set_yticklabels([str(i) for i in intbits])
                        ax.set_xlabel('n_bits')
                        ax.set_ylabel('n_intbits')
                        ax.set_title(key + " c={}, {}, np: {:.4f}".format(c,
                                                                          approx,
                                                                          bl))

                        thresh = 0.5
                        for i in range(len(intbits)):
                            for j in range(len(bits)):
                                val = f[i, j]
                                color = 'w' if val < thresh else 'k'
                                ax.annotate("{:.4f}".format(val),
                                            xy=(j, i),
                                            horizontalalignment='center',
                                            verticalalignment='center',
                                            color=color)
                fig.suptitle("{} + {}".format(opt, str(approx)), fontsize=20)
                fig.show()
        return fig

    def plot_success_for(self, approximation, constraint):
        """Plot for which fp precisions the run was successful."""
        intbits = self.n_intbits_range
        bits = self.n_bits_range
        f = np.zeros((len(intbits), len(bits)))
        r = self.results
        for i, ib in enumerate(intbits):
            for j, b in enumerate(bits):
                f[i, j] = r.loc[(r['constraint'] == constraint) &
                                (r['approximation'] == approximation) &
                                (r['n_bits'] == b) &
                                (r['n_intbits'] == ib), 'failed']
        plt.figure()
        plt.imshow(1 - f)
        plt.xticks(range(len(bits)), [str(i) for i in bits])
        plt.yticks(range(len(intbits)), [str(i) for i in intbits])
        plt.grid()
        plt.colorbar()
        plt.xlabel('n_bits')
        plt.ylabel('n_intbits')
        plt.title("Successful (1) vs failed (0) runs for "
                  "approx={} and c={}".format(approximation, constraint))

    def _get_values(self, key):
        """Get distinct values for a certain property."""
        return sorted(list(self.results[key].value_counts().index.values))

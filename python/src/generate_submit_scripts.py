"""
Generate the submit script for a large number of single runs to be submitted
to a cluster with a condor scheduler.
"""

import os

import numpy as np

from spfpm.FixedPoint import FXfamily


executable = '/path/to/python/bin/python3'
base = '/path/to/output/folder/'

datasets = ['adult', 'synthetic', 'compas', 'bank', 'german', 'sqf']
optimizers = ['lagrange', 'iplb', 'projected']
fairness = ['ppercent']
approximations = ['none', 'secureml']
bits = [(None, None), (16, 16)]
constraints = list(np.logspace(-4, 0, 25))
# adult 32768, bank 32768, compas 4096, german 512, sqf 65536, synthetic 4096
epochs = {'adult': int(np.round(960000/32768)),
          'bank': int(np.round(960000/32768)),
          'compas': int(np.round(960000/4096)),
          'german': 500,
          'sqf': int(np.round(960000/65536)),
          'synthetic': int(np.round(960000/4096))}
learning_rate_decay = False
batchsizes = [64]
max_sample_size = -1
update_rule = 'vanilla'
verbose = 1
random_state = 111

lines = list()
lines.append('executable = {}'.format(executable))
lines.append('')
lines.append('request_memory = 512')
lines.append('request_cpus = 1')
lines.append('getenv = True')
lines.append('')

counter = 0

baseline_tracker = set()

for dataset in datasets:
    d1 = os.path.abspath(os.path.join(base, dataset))
    for fair in fairness:
        d2 = os.path.join(d1, fair)
        for approx in approximations:
            d3 = os.path.join(d2, approx)
            for constraint in constraints:
                d4 = os.path.join(d3, str(constraint))
                for optimizer in optimizers:
                    d5 = os.path.join(d4, optimizer)
                    if not os.path.exists(d5):
                        os.makedirs(d5)
                    for batchsize in batchsizes:
                        fbase = os.path.join(d5,
                                             'bs={}'.format(str(batchsize)))
                        for nint, nfrac in bits:
                            if nint is None or nfrac is None:
                                family = None
                                nint = 0
                                nfrac = 0
                                fn = fbase
                            else:
                                family = FXfamily(n_bits=nfrac,
                                                  n_intbits=nint)
                                fn = fbase + '_bits={}.{}'.format(nint,
                                                                  nfrac)

                            current_baseline = dataset + str(constraint)
                            if fair == 'ppercent' and current_baseline not in baseline_tracker:
                                baseline_tracker.add(current_baseline)
                                d_bl = os.path.abspath(os.path.join(base,
                                                                    dataset,
                                                                    fair,
                                                                    str(constraint),
                                                                    'baseline'))
                                os.makedirs(d_bl)
                                fn_bl = os.path.join(d_bl, 'bl')
                                lines.append('output = ' + fn_bl + '.out')
                                lines.append('error = ' + fn_bl + '.err')
                                lines.append('log = ' + fn_bl + '.log')
                                command = ('quick_run.py ' +
                                           '--outdir {} ' +
                                           '--optimizer baseline ' +
                                           '--dataset {} ' +
                                           '--fairness {} ' +
                                           '--constraint {} ' +
                                           '--max-sample-size {} ').format(base,
                                                                   dataset,
                                                                   fair,
                                                                   constraint,
                                                                   max_sample_size)

                                lines.append('arguments="' + command + '"')
                                lines.append('queue')
                                lines.append('')
                                counter += 1

                            lines.append('output = ' + fn + '.out')
                            lines.append('error = ' + fn + '.err')
                            lines.append('log = ' + fn + '.log')
                            command = ('quick_run.py ' +
                                       '--outdir {} ' +
                                       '--optimizer {} ' +
                                       '--dataset {} ' +
                                       '--fairness {} ' +
                                       '--approximation {} ' +
                                       '--constraint {} ' +
                                       '--epochs {} ' +
                                       '--batchsize {} ' +
                                       '--max-sample-size {} ' +
                                       '--verbose {} ' +
                                       '--random-state {} ' +
                                       '--nbits {} ' +
                                       '--nintbits {}').format(base,
                                                               optimizer,
                                                               dataset,
                                                               fair,
                                                               approx,
                                                               constraint,
                                                               epochs[dataset],
                                                               batchsize,
                                                               max_sample_size,
                                                               verbose,
                                                               random_state,
                                                               nfrac,
                                                               nint)
                            if family is None:
                                command += ' --floats'
                            if learning_rate_decay:
                                command += ' --learning-rate-decay'
                            else:
                                command += ' --no-learning-rate-decay'
                            if optimizer == 'projected':
                                command += ' --update-rule vanilla'
                            else:
                                command += ' --update-rule ' + update_rule

                            lines.append('arguments="' + command + '"')
                            lines.append('queue')
                            lines.append('')
                            counter += 1

with open('run_all.sub', 'w') as file:
    file.write('\n'.join(lines))

print('Submit file with {} jobs created!'.format(counter))

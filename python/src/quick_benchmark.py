"""
This script runs a quick benchmark for FairLogisticRegression on a given dataset
"""

from benchmark_flr import BenchmarkFLR
from data import *
from fair_logistic_regression import FairLogisticRegression


# Set the dataset and the fairness notion
dataset = 'adult'  # 'adult', 'compas', 'synthetic', 'german', 'bank', 'sqf'
fairness = 'ppercent'

# Get the dataset
if dataset == 'adult':
    Xtr, Xte, ytr, yte, Ztr, Zte = read_adult()
elif dataset == 'compas':
    Xtr, Xte, ytr, yte, Ztr, Zte = read_compas()
elif dataset == 'synthetic':
    Xtr, Xte, ytr, yte, Ztr, Zte = read_synthetic()
elif dataset == 'bank':
    Xtr, Xte, ytr, yte, Ztr, Zte = read_bank()
elif dataset == 'german':
    Xtr, Xte, ytr, yte, Ztr, Zte = read_german()
elif dataset == 'sqf':
    Xtr, Xte, ytr, yte, Ztr, Zte = read_sqf()
else:
    raise ValueError("Dataset {} is unknown".format(dataset))

# Only ppercent implemented in this repository
if fairness == 'ppercent':
    fairness_model = FairLogisticRegression
else:
    raise ValueError("Fairness notion {} is unknown".format(fairness))

# Input the different settings to run directly here
bm = BenchmarkFLR(optimizations=['unconstrained', 'lagrange', 'iplb', 'projected'],
                  approximations=['none', 'piecewise', 'secureml'],
                  update_rule='momentum',  # 'vanilla', 'nesterov'
                  n_bits_range=[16],
                  n_intbits_range=[16],
                  constraints=[1., 0.1, 0.01],
                  batchsize=128,
                  epochs=20)

bm.run_benchmark(Xtr, ytr, Ztr, Xte, yte)

bm.plot_accuracies()

"""
This short script traverses down the directory tree and combines the results in
all found pkl files together with their argument settings and combines
everything into a big csv table for final analysis of multiple runs.
"""

import argparse
import glob
import os
import pickle

import pandas as pd


# Setup argparse
formatter_class = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter_class)
parser.add_argument('--directory',
                    help='Directory in which to search for result pkl files',
                    type=str,
                    default='/is/cluster/nkilbertus/fair-mpc/results')
parser.add_argument('--out',
                    help='Filename where to save result',
                    type=str,
                    default='collection.csv')

# Get arguments
args = parser.parse_args().__dict__
basedir = args['directory']
output = args['out']


# Find result files
os.chdir(basedir)
paths = glob.iglob('**/*_args.pkl', recursive=True)
assert paths, "Did not find any pkl files in {}".format(basedir)

# Collect results pkl files
print("Collecting results ...")
results = pd.DataFrame()
for i, path in enumerate(paths):
    path = os.path.abspath(path)
    with open(path, 'rb') as file:
        args = pickle.load(file)
    res_path = path.replace('_args', '')
    with open(res_path, 'rb') as file:
        result = pickle.load(file)
    results = results.append({**args, **result}, ignore_index=True)
    if i % 100 == 0:
        print('processed: ', i)
print("DONE")

# Write collected results to csv file
print("Writing results dataframe to {} ...".format(output), end=' ')
results.to_csv(output)
print("DONE")

"""
This module contains all data loading functionality.
"""

import os
import pdb

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal  # generating synthetic data


# ------------------------------------------------------------------------------
# Read Different Datasets
# ------------------------------------------------------------------------------

train_frac = 0.8


# ------------------------------------------------------------------------------
# german.data-numeric
# age is sensitive attribute as in:
# 1. https://www.cs.toronto.edu/~toni/Papers/icml-final.pdf
# 2. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.182.6067&rep=rep1&type=pdf
# age > 25 = old
# age <=25 = young
# for now pretend everything is numeric because numeric dataset is
# somehow significantly different from non-numeric dataset
# (e.g., see column 4)
def read_german(filepath='../data/german.data-numeric',
                random_state=42):
    """
    Read the german dataset.

    The training test set split is set to 0.8, but the actual training set size
    is the largest power of two smaller than 0.8 * n_examples and the actual
    test set is everything that remains.

    Args:
        filepath: The file path to the data
        random_state: The random seed of the train-test split

    Returns:
        Xtr, Xte, ytr, yte, Ztr, Zte: X are the features, y are the labels and Z
            are the sensitive attributes. 'tr' stands for training and 'te' for
            test set.
    """
    if filepath.endswith('ready.npz') and os.path.exists(filepath):
        return load_npz(filepath)
    df = pd.read_csv(filepath, header=None, delim_whitespace=True)

    # change label to 0/1
    cols = list(df)
    label_idx = len(cols)-1
    df[label_idx] = df[label_idx].map({2: 0, 1: 1})

    M = df.as_matrix()
    Z = M[:, 9]
    Z = (Z > 25).astype(float).reshape(-1, 1)
    ix = np.delete(np.arange(24), 9)
    X = M[:, ix]
    y = M[:, -1]

    n = X.shape[0]  # Number of examples

    # Create train test split
    tr_idx, te_idx = _get_train_test_split(n, train_frac, random_state)
    Xtr, Xte, ytr, yte, Ztr, Zte = _apply_train_test_split(X, y, Z,
                                                           tr_idx, te_idx)

    # Whiten feature data
    Xtr, Xte = _whiten_data(Xtr, Xte)

    # Center sensitive data
    Ztr, Zte = _center_data(Ztr, Zte)

    # Add intercept
    Xtr, Xte = _add_intercept(Xtr, Xte)

    # Labels are already 0/1

    return Xtr, Xte, ytr, yte, Ztr, Zte


# ------------------------------------------------------------------------------
# compas_preprocessed.npz
def read_compas(filepath='../data/compas_preprocessed.npz',
                random_state=42):
    """
    Read the COMPAS dataset.

    Args:
        filepath: The file path to the data
        random_state: The random seed for train test split

    Returns:
        Xtr, Xte, ytr, yte, Ztr, Zte: X are the features, y are the labels and Z
            are the sensitive attributes. 'tr' stands for training and 'te' for
            test set.
    """
    if filepath.endswith('ready.npz') and os.path.exists(filepath):
        return load_npz(filepath)

    d = np.load(filepath)
    X, y, Z = d['X'], d['y'], d['Z']

    n = X.shape[0]  # Number of examples

    # Get train test split
    tr_idx, te_idx = _get_train_test_split(n, train_frac, random_state)
    Xtr, Xte, ytr, yte, Ztr, Zte = _apply_train_test_split(X, y, Z,
                                                           tr_idx, te_idx)

    # Whiten feature data
    Xtr, Xte = _whiten_data(Xtr, Xte)

    # Center sensitive data
    Ztr, Zte = _center_data(Ztr, Zte)

    # Add intercept
    Xtr, Xte = _add_intercept(Xtr, Xte)

    # Labels are already 0/1

    return Xtr, Xte, ytr, yte, Ztr, Zte


# ------------------------------------------------------------------------------
# compas_all_data.csv
def deprecated_read_compas(filepath='../data/compas_all_data.csv',
                           random_state=42):
    """
    Read the COMPAS dataset.

    Args:
        filepath: The file path to the data
        random_state: The random seed for train test split

    Returns:
        Xtr, Xte, ytr, yte, Ztr, Zte: X are the features, y are the labels and Z
            are the sensitive attributes. 'tr' stands for training and 'te' for
            test set.
    """
    if filepath.endswith('ready.npz') and os.path.exists(filepath):
        return load_npz(filepath)

    f0_race = [
        'Caucasian',
        'African-American',
        'Asian',
        'Hispanic',
        'Native American',
        'Other']

    # read in data as floating point
    d = 6
    p = 7
    X = np.zeros((0, d))
    Z = np.zeros((0, p))
    y = np.zeros((0,))
    with open(filepath, 'r') as f:
        count = 0
        for line in f:
            line = line.strip()
            parts = line.split(',')
            feats = np.array([float(parts[2])] +
                             [float(parts[3][1:-1] == "F")] +
                             [float(parts[4])] +
                             [float(parts[5])] +
                             [float(parts[6])] +
                             [float(parts[7])], dtype=np.float32)
            sense = np.array(_onek_encoding_unk(parts[0][1:-1], f0_race) +
                             [float(parts[1][1:-1] == "Female")], dtype=np.float32)
            label = np.array(float(parts[8]), dtype=np.float32)  # 0,1
            X = np.concatenate((X, feats.reshape(1, d)), 0)
            Z = np.concatenate((Z, sense.reshape(1, p)), 0)
            y = np.concatenate((y, label.reshape(1,)), 0)
            count = count + 1

    n = X.shape[0]  # Number of examples

    # Get train test split
    tr_idx, te_idx = _get_train_test_split(n, train_frac, random_state)
    Xtr, Xte, ytr, yte, Ztr, Zte = _apply_train_test_split(X, y, Z,
                                                           tr_idx, te_idx)

    # Whiten feature data
    Xtr, Xte = _whiten_data(Xtr, Xte)

    # Center sensitive data
    Ztr, Zte = _center_data(Ztr, Zte)

    # Add intercept
    Xtr, Xte = _add_intercept(Xtr, Xte)

    # Labels are already 0/1

    return Xtr, Xte, ytr, yte, Ztr, Zte


# ------------------------------------------------------------------------------
# bank-additional-full.csv
def read_bank(filepath='../data/bank-additional-full.csv',
              random_state=42):
    """
    Read the UCI Bank dataset.

    Args:
        filepath: The file path to the data
        random_state: The random seed for train test split

    Returns:
        Xtr, Xte, ytr, yte, Ztr, Zte: X are the features, y are the labels and Z
            are the sensitive attributes. 'tr' stands for training and 'te' for
            test set.
    """
    if filepath.endswith('ready.npz') and os.path.exists(filepath):
        return load_npz(filepath)
    f1_job = [
        'admin.',
        'blue-collar',
        'entrepreneur',
        'housemaid',
        'management',
        'retired',
        'self-employed',
        'services',
        'student',
        'technician',
        'unemployed',
        'unknown']
    f2_marital = [
        'divorced',
        'married',
        'single',
        'unknown']
    f3_education = [
        'basic.4y',
        'basic.6y',
        'basic.9y',
        'high.school',
        'illiterate',
        'professional.course',
        'university.degree',
        'unknown']
    f4_default = [
        'no',
        'yes',
        'unknown']
    f5_housing = [
        'no',
        'yes',
        'unknown']
    f6_loan = [
        'no',
        'yes',
        'unknown']
    f8_month = [
        'jan',
        'feb',
        'mar',
        'apr',
        'may',
        'jun',
        'jul',
        'aug',
        'sep',
        'oct',
        'nov',
        'dec']
    f9_day = [
        'mon',
        'tue',
        'wed',
        'thu',
        'fri']
    f14_poutcome = [
        'failure',
        'nonexistent',
        'success']

    # read in data as floating point
    # we won't use 2 features:
    # 11 (10 below) - duration: it is not known if a call is performed (and if equal to 0 y=0)
    # 13 (12 below) - pdays: it is very often 999 (client was not previously contacted)
    # TODO: adopted this hacky way to read in the dataset, can be improved
    i = -1
    with open(filepath) as f:
        for i, _ in enumerate(f):
            pass
    n_lines = i + 1
    d = 61
    X = np.zeros((n_lines, d))
    Z = np.zeros((n_lines, 1))
    y = np.zeros((n_lines,))
    with open(filepath, 'r') as f:
        count = 0
        for i, line in enumerate(f):
            line = line.strip()
            parts = line.split(';')
            feats = np.array(_onek_encoding_unk(parts[1][1:-1], f1_job) +
                             _onek_encoding_unk(parts[2][1:-1], f2_marital) +
                             _onek_encoding_unk(parts[3][1:-1], f3_education) +
                             _onek_encoding_unk(parts[4][1:-1], f4_default) +
                             _onek_encoding_unk(parts[5][1:-1], f5_housing) +
                             _onek_encoding_unk(parts[6][1:-1], f6_loan) +
                             [float(parts[7][1:-1] == 'cellular')] +
                             _onek_encoding_unk(parts[8][1:-1], f8_month) +
                             _onek_encoding_unk(parts[9][1:-1], f9_day) +
                             [float(parts[11])] +
                             [float(parts[13])] +
                             _onek_encoding_unk(parts[14][1:-1], f14_poutcome) +
                             [float(parts[15])] +
                             [float(parts[16])] +
                             [float(parts[17])] +
                             [float(parts[18])] +
                             [float(parts[19])], dtype=np.float32)
            sense = np.array([float((int(parts[0]) < 25) | (int(parts[0]) > 60))],
                             dtype=np.float32)
            label = np.array((float(parts[20][1:-1] == 'yes')),
                             dtype=np.float32)  # 0,1
            X[i, :] = feats
            Z[i, :] = sense
            y[i] = label
            count = count + 1

    n = X.shape[0]  # Number of examples

    # Get train test split
    tr_idx, te_idx = _get_train_test_split(n, train_frac, random_state)
    Xtr, Xte, ytr, yte, Ztr, Zte = _apply_train_test_split(X, y, Z,
                                                           tr_idx, te_idx)

    # Whiten feature data
    Xtr, Xte = _whiten_data(Xtr, Xte)

    # Center sensitive data
    Ztr, Zte = _center_data(Ztr, Zte)

    # Add intercept
    Xtr, Xte = _add_intercept(Xtr, Xte)

    # Labels are already 0/1

    return Xtr, Xte, ytr, yte, Ztr, Zte


# ------------------------------------------------------------------------------
def read_adult(filepath='../data/adult.npz',
               random_state=42):
    """
    Read the UCI Adult dataset.

    Args:
        filepath: The file path to the data
        random_state: The random seed for train test split

    Returns:
        Xtr, Xte, ytr, yte, Ztr, Zte: X are the features, y are the labels and Z
            are the sensitive attributes. 'tr' stands for training and 'te' for
            test set.
    """
    if filepath.endswith('ready.npz') and os.path.exists(filepath):
        return load_npz(filepath)
    data = np.load(filepath)

    X = np.concatenate((data['Xtr'][:, 1:], data['Xte'][:, 1:]))
    y = np.concatenate((data['ytr'], data['yte']))
    Z = np.concatenate((data['Ztr'], data['Zte']))
    Z = Z[:, np.newaxis]
    n = X.shape[0]  # Number of examples

    # Get train test split
    tr_idx, te_idx = _get_train_test_split(n, train_frac, random_state)
    Xtr, Xte, ytr, yte, Ztr, Zte = _apply_train_test_split(X, y, Z,
                                                           tr_idx, te_idx)

    # Whiten feature data
    Xtr, Xte = _whiten_data(Xtr, Xte)

    # Center sensitive data
    Ztr, Zte = _center_data(Ztr, Zte)

    # Use 0 and 1 labels
    _labels_to_zero_one(ytr, yte)

    # Add back a column of 1s for bias terms
    Xtr, Xte = _add_intercept(Xtr, Xte)

    return Xtr, Xte, ytr, yte, Ztr, Zte


# ------------------------------------------------------------------------------
def read_sqf(filepath='../data/stop_and_frisk_2012.npz',
             random_state=42):
    """
    Read the stop, question, and frisk (SQF) 2012 dataset.

    Args:
        filepath: The file path to the data
        random_state: The random seed for train test split

    Returns:
        Xtr, Xte, ytr, yte, Ztr, Zte: X are the features, y are the labels and Z
            are the sensitive attributes. 'tr' stands for training and 'te' for
            test set.
    """
    if filepath.endswith('ready.npz') and os.path.exists(filepath):
        return load_npz(filepath)

    data = np.load(filepath)
    X, y, Z = data['arr_0'], data['arr_1'], data['arr_2']
    Z = Z[:, np.newaxis]

    n = X.shape[0]  # Number of examples

    # Get train test split
    tr_idx, te_idx = _get_train_test_split(n, train_frac, random_state)
    Xtr, Xte, ytr, yte, Ztr, Zte = _apply_train_test_split(X, y, Z,
                                                           tr_idx, te_idx)

    # Whiten feature data
    Xtr, Xte = _whiten_data(Xtr, Xte)

    # Center sensitive data
    Ztr, Zte = _center_data(Ztr, Zte)

    # Add intercept
    Xtr, Xte = _add_intercept(Xtr, Xte)

    # Convert labels are already 0/1
    _labels_to_zero_one(ytr, yte)

    return Xtr, Xte, ytr, yte, Ztr, Zte


# ------------------------------------------------------------------------------
# This has been adopted from Zafar et al. at
# https://github.com/mbilalzafar/fair-classification
def read_synthetic(n_samples=4096, disc_factor=np.pi/8., random_state=42):
    """
    Code for generating the synthetic data.

    We will have two non-sensitive features and one sensitive feature.
    A sensitive feature value of 0.0 means the example is considered to be
    in protected group (e.g., female) and 1.0 means it's in non-protected
    group (e.g., male).

    Args:
        n_samples: Number of examples
        disc_factor: Discrimination factor
        random_state: The random seed

    Returns:
        Xtr, Xte, ytr, yte, Ztr, Zte: X are the features, y are the labels and Z
            are the sensitive attributes. 'tr' stands for training and 'te' for
            test set.
    """
    def gen_gaussian(mean_in, cov_in, class_label):
        nv = multivariate_normal(mean=mean_in, cov=cov_in)
        x_tmp = nv.rvs(n_samples)
        y_tmp = np.ones(n_samples, dtype=float) * class_label
        return nv, x_tmp, y_tmp

    # Generate the non-sensitive features randomly
    # We will generate one gaussian cluster for each class
    mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
    mu2, sigma2 = [-2, -2], [[10, 1], [1, 3]]
    nv1, X1, y1 = gen_gaussian(mu1, sigma1, 1)  # positive class
    nv2, X2, y2 = gen_gaussian(mu2, sigma2, -1)  # negative class

    # join the posisitve and negative class clusters
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    # shuffle the data
    perm = list(range(0, n_samples*2))
    np.random.shuffle(perm)
    X = X[perm]
    y = y[perm].astype(int)

    rotation_mult = np.array([[np.cos(disc_factor), -np.sin(disc_factor)],
                              [np.sin(disc_factor), np.cos(disc_factor)]])
    X_aux = np.dot(X, rotation_mult)

    # Generate the sensitive feature here
    # this array holds the sensitive feature value
    Z = []
    for i in range(0, len(X)):
        x = X_aux[i]

        # probability for each cluster that the point belongs to it
        p1 = nv1.pdf(x)
        p2 = nv2.pdf(x)

        # normalize the probabilities from 0 to 1
        s = p1 + p2
        p1 = p1 / s

        # generate a random number from 0 to 1
        r = np.random.uniform()

        # the first cluster is the positive class
        if r < p1:
            Z.append(1.0)  # 1.0 -> male
        else:
            Z.append(0.0)  # 0.0 -> female

    Z = np.array(Z)[:, np.newaxis]
    n = X.shape[0]  # Number of examples
    p = 0.8  # Training set fraction

    # Get train test split
    tr_idx, te_idx = _get_train_test_split(n, p, random_state)
    Xtr = X[tr_idx, :]
    Xte = X[te_idx, :]
    ytr = y[tr_idx]
    yte = y[te_idx]
    Ztr = Z[tr_idx, :]
    Zte = Z[te_idx, :]

    # Whiten feature data
    Xtr, Xte = _whiten_data(Xtr, Xte)

    # Center sensitive data
    Ztr, Zte = _center_data(Ztr, Zte)

    # Add intercept
    Xtr, Xte = _add_intercept(Xtr, Xte)

    # Convert labels to 0/1
    _labels_to_zero_one(ytr, yte)

    return Xtr, Xte, ytr, yte, Ztr, Zte


# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------


def export_npz_files(subsample=False):
    """Call all read data functions and export the data as npz files."""
    for func in [read_adult, read_bank, read_compas, read_german, read_sqf]:
        print("Export {} dataset ...".format(func.__name__), end=' ')
        xtr, xte, ytr, yte, ztr, zte = func()
        filename = func.__name__.replace('read_', '') + '_ready.npz'
        # do we subsample training data to have power of 2 examples?
        n = xtr.shape[0]
        if subsample and ((n & (n - 1)) and n != 0):
            # The largest power of 2 smaller than n
            nmax = 2**(len(bin(n)) - 3)
            xtr = xtr[:nmax, :]
            ytr = ytr[:nmax]
            ztr = ztr[:nmax, :]
        with open(filename, 'wb') as handle:
            np.savez(handle, Xtr=xtr, Xte=xte, ytr=ytr.astype(int),
                     yte=yte.astype(int), Ztr=ztr, Zte=zte)
        print("DONE")


def print_dataset_sizes(subsample=False):
    """Print a little summary of all datasets."""
    for func in [read_adult, read_bank, read_compas, read_german, read_sqf,
                 read_synthetic]:
        xtr, xte, ytr, yte, ztr, zte = func()
        if subsample:
            nmax = 1 << (xtr.shape[0].bit_length() - 1)
            xtr = xtr[:nmax, :]
            ytr = ytr[:nmax]
            ztr = ztr[:nmax, :]
        ntr = xtr.shape[0]
        nte = xte.shape[0]
        print("--- {} ---".format(func.__name__))
        print('n tr: ', ntr)
        print('n te: ', nte)
        print('d: ', xtr.shape[1])
        print('p: ', ztr.shape[1])
        print('y=1 tr/te: {:.3f} / {:.3f}'.format(sum(ytr) / ntr,
                                                  sum(yte) / nte))
        for j in range(ztr.shape[1]):
            print('z' + str(j) + '=1 {:.3f} / {:.3f}'.format(
                  sum(ztr[:, j] > 0) / ntr,
                  sum(zte[:, j] > 0) / nte))
        print('\n')


def data_checks():
    """Run some simple checks on the datasets."""
    for func in [read_adult, read_bank, read_compas, read_german, read_sqf,
                 read_synthetic]:
        xtr, xte, ytr, yte, ztr, zte = func()

        if np.any(xtr[:, 0] != 1.) or np.any(xte[:, 0] != 1.):
            print("WARNING: intercept issue in {}".format(func.__name__))
        if np.any((ytr != 1) & (ytr != 0)) or np.any((yte != 1) & (yte != 0)):
            print("WARNING: label issue in {}".format(func.__name__))
        if np.any(np.std(xtr[:, 1:], 0) == 0) or np.any(np.std(xte[:, 1:], 0) == 0):
            print("WARNING: constant column in X {}".format(func.__name__))
        if np.any(np.std(ztr, 0) == 0) or np.any(np.std(zte, 0) == 0):
            print("WARNING: constant column in Z {}".format(func.__name__))
        if np.std(ytr) == 0 or np.std(yte) == 0:
            print("WARNING: constant column in y {}".format(func.__name__))

    print("Done running checks.")


# ------------------------------------------------------------------------------
# Internal helper Functions
# ------------------------------------------------------------------------------


def _add_intercept(train, test):
    """Add a column of all ones for the intercept."""
    return tuple([np.hstack((np.ones((x.shape[0], 1)), x))
                  for x in (train, test)])


def _labels_to_zero_one(*args):
    """Map labels to take values in zero and one."""
    for x in args:
        x[x <= 0.] = 0
        x[x > 0.] = 1


def _labels_to_plus_minus(*args):
    """Map labels to take values in minus one and one."""
    for x in args:
        x[x <= 0.] = -1
        x[x > 0.] = 1


def _shuffle_data(*args, random_state=42):
    """Shuffle data with random permutation."""
    n = args[0].shape[0]
    np.random.seed(random_state)
    perm = np.random.permutation(n)
    return tuple([x[perm, :] if x.ndim > 1 else x[perm] for x in args])


def _center_data(train, test):
    """Center the data, i.e. subtract the mean column wise."""
    mean = np.mean(train, 0)
    return train - mean, test - mean


def _whiten_data(train, test):
    """Whiten training and test data with training mean and std dev."""
    mean = np.mean(train, 0)
    std = np.std(train, 0)
    return tuple([(x - mean) / (std + 1e-7) for x in (train, test)])


def _get_train_test_split(n_examples, train_fraction, seed, power_of_two=True):
    """
    Args:
        n_examples: Number of training examples
        train_fraction: Fraction of data to use for training
        seed: Seed for random number generation (reproducability)
        power_of_two: Whether to select the greatest power of two for training
            set size and use the remaining examples for testing.

    Returns:
        training indices, test indices
    """
    np.random.seed(seed)
    idx = np.random.permutation(n_examples)
    pivot = int(n_examples * train_fraction)
    if power_of_two:
        pivot = 2**(len(bin(pivot)) - 3)
    training_idx = idx[:pivot]
    test_idx = idx[pivot:]
    return training_idx, test_idx


def _apply_train_test_split(x, y, z, training_idx, test_idx):
    """
    Apply the train test split to the data.

    Args:
        x: Features
        y: Labels
        z: Sensitive attributes
        training_idx: Training set indices
        test_idx: Test set indices

    Returns:
        Xtr, Xte, ytr, yte, Ztr, Zte: X are the features, y are the labels and Z
            are the sensitive attributes. 'tr' stands for training and 'te' for
            test set.
    """
    Xtr = x[training_idx, :]
    Xte = x[test_idx, :]
    ytr = y[training_idx]
    yte = y[test_idx]
    Ztr = z[training_idx, :]
    Zte = z[test_idx, :]
    return Xtr, Xte, ytr, yte, Ztr, Zte


def _onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        print('not in allowable!')
        pdb.set_trace()
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def load_npz(filepath):
    """Load a preprocessed dataset that has been saved in numpy's npz format."""
    d = np.load(filepath)
    return d['Xtr'], d['Xte'], d['ytr'], d['yte'], d['Ztr'], d['Zte']

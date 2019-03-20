"""
This is a small script to demonstrate that an unconstrained but strongly
regularized logistic regression classifier can achieve high accuracy while
fulfilling almost any ppercent constraint, no matter how tight the constraint
value.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from data import *


def check_baseline(dataset, C=1e-8):
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

    # Fit an unconstrained baseline on floats
    baseline = LogisticRegression(C=C, random_state=42)
    baseline.fit(Xtr, ytr)

    train_acc = np.mean(ytr == baseline.predict(Xtr))
    test_acc = np.mean(yte == baseline.predict(Xte))

    w = np.array(baseline.coef_[0])
    c_vals = np.abs(Ztr.T @ Xtr @ w / Xtr.shape[0])
    print("---- {} ----".format(dataset))
    print("Train Accuracy: ", train_acc)
    print("Test Accuracy: ", test_acc)
    print("Max Abs Constr. Value: ", np.max(c_vals))

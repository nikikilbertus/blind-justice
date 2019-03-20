"""
This script performs a single training run for a specific parameter setting.
"""

import os
import pickle
import numpy as np
from copy import deepcopy
from pprint import pprint
from timeit import default_timer as timer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from data import *
from fair_logistic_regression import FairLogisticRegression
from ppercent_baseline import PpercentBaseline
from quick_run_parser import QuickRunParser
from spfpm.FixedPoint import FXfamily
from tools import *


# -----------------------------------------------------------------------------
# A single run called from command line
# -----------------------------------------------------------------------------


def store_weight_independent_results(result_dict, model):
    result_dict['final_lrw'] = to_float(model.lrw)
    result_dict['final_lrl'] = to_float(model.lrl)
    result_dict['final_lambda'] = to_float(model.lam)
    idx = index_of_last_non_zero(model.history['constraint_satisfied'])
    result_dict['last_valid_epoch'] = idx


def get_fairness_model(fairness_notion, args):
    if fairness_notion == 'ppercent':
        if args['optimizer'] == 'baseline':
            fairness_model = PpercentBaseline
        else:
            fairness_model = FairLogisticRegression
    else:
        raise ValueError("Fairness notion {} is unknown".format(fairness_notion))
    return fairness_model


def monitoring_print(n, d, p, nte, dte, pte, family, args):
    print("---- Dataset ----")
    print(args['dataset'])
    print("\n")
    print("---- Training Dimensions ----")
    print("n: ", n)
    print("d: ", d)
    print("p: ", p)
    print("\n")
    print("---- Test Dimensions ----")
    print("n: ", nte)
    print("d: ", dte)
    print("p: ", pte)
    print("\n")
    print("---- Fairness Notion ----")
    print(args['fairness'])
    print("\n")
    print("---- Constraint ----")
    print(args['constraint'])
    print("\n")
    print("---- Optimizer ----")
    print(args['optimizer'])
    print("\n")
    if args['optimizer'] != 'baseline':
        print("---- Approximation ----")
        print(args['approximation'])
        print("\n")
        print("---- Number format ----")
        if family is None:
            print("floats")
        else:
            print(args['nbits'], args['nintbits'])
    print("\nNumber of epochs to ", args['epochs'])
    print("\n", flush=True)


if __name__ == '__main__':

    # --------------------------------------------------------------------------
    # Parse the arguments
    parser = QuickRunParser()
    args = parser.parse_args()
    if args['verbose'] > 0:
        pprint(args)

    # --------------------------------------------------------------------------
    # Set the seed
    np.random.seed(args['random_state'])

    # --------------------------------------------------------------------------
    # Get the dataset
    dataset = args['dataset']
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

    # --------------------------------------------------------------------------
    # If we are running the "exact" baseline, remove unused arguments
    # This makes sure we do not actually later rely on them
    if args['optimizer'] == 'baseline':
        assert args['fairness'] == 'ppercent', \
            "Basline optimizer only available for ppercent."

    # Get the number family
    if args['floats']:
        family = None
    else:
        family = FXfamily(args['nbits'], args['nintbits'])

    # --------------------------------------------------------------------------
    # Always subsample training data to have power of 2 examples or set maximum
    # The largest power of 2 smaller than n
    n = Xtr.shape[0]
    if not is_pow2(n):
        n = 1 << (n.bit_length() - 1)
        print("Training set did not have power of 2 examples! Subsampling!")
    args_nmax = args['max_sample_size']
    if args_nmax > 0:
        # Check if chosen max sample size is power of two
        if is_pow2(args_nmax):
            n = min(n, args_nmax)
        else:
            raise ValueError("Max sample size {} must be power of two"
                             .format(args_nmax))
    if n != Xtr.shape[0]:
        Xtr = Xtr[:n, :]
        ytr = ytr[:n]
        Ztr = Ztr[:n, :]

    # --------------------------------------------------------------------------
    # Which fairness notion to use (only ppercent implemented in this repo)
    fairness = args['fairness']
    fairness_model = get_fairness_model(fairness, args)

    # --------------------------------------------------------------------------
    # Overwrite number of epochs if chosen
    if args['gradient_updates'] > 0:
        args['epochs'] = int(args['gradient_updates'] * args['batchsize'] / n)

    # --------------------------------------------------------------------------
    # Get the shapes of inputs
    n = Xtr.shape[0]
    d = Xtr.shape[1]
    p = Ztr.shape[1]
    nte = Xte.shape[0]
    dte = Xte.shape[1]
    pte = Zte.shape[1]

    # --------------------------------------------------------------------------
    # Some prints for monitoring
    monitoring_print(n, d, p, nte, dte, pte, family, args)

    # --------------------------------------------------------------------------
    # Fit an unconstrained baseline on floats
    unconstrained = LogisticRegression(C=1e4, random_state=args['random_state'])
    unconstrained.fit(Xtr, ytr)
    baseline_accuracy = np.mean(yte == unconstrained.predict(Xte))

    # --------------------------------------------------------------------------
    # Fit the fair model
    if args['optimizer'] != 'baseline':
        model = fairness_model(opt=args['optimizer'],
                               approx=args['approximation'],
                               family=family,
                               constraint=args['constraint'],
                               update_rule=args['update_rule'],
                               momentum=args['momentum'],
                               weight_decay=args['weight_decay'],
                               learning_rate=args['learning_rate'],
                               learning_rate_lambda=args['learning_rate_lambda'],
                               lambda_fixed=args['lambda_fixed'],
                               learning_rate_decay=args['learning_rate_decay'],
                               verbose=args['verbose'],
                               random_state=args['random_state'])

        start = timer()
        model.fit(Xtr, ytr, Ztr,
                  epochs=args['epochs'],
                  batchsize=args['batchsize'],
                  Xval=Xte,
                  yval=yte)
        runtime = timer() - start
    else:
        model = fairness_model(constraint=args['constraint'])
        start = timer()
        model.fit(Xtr, ytr, Ztr, Xval=Xte, yval=yte, Zval=Zte)
        runtime = timer() - start

    # --------------------------------------------------------------------------
    # Keep track of weight and model independent results
    results = {
        'n_training': n,
        'd_training': d,
        'p_training': p,
        'n_test': nte,
        'd_test': dte,
        'p_test': pte,
        'runtime': runtime,
        'baseline_accuracy': baseline_accuracy
    }

    # --------------------------------------------------------------------------
    # Keep track of weight independent results
    if args['optimizer'] != 'baseline':
        store_weight_independent_results(results, model)

    # --------------------------------------------------------------------------
    # Print a weight independent summary
    print("---- Unconstrained Baseline on floats ----")
    print("Accuracy: ", baseline_accuracy)
    # print(classification_report(yte, baseline.predict(Xte)))
    print("\n")

    # --------------------------------------------------------------------------
    # Get the weights for the model
    if args['optimizer'] != 'baseline':
        weights = {'final': deepcopy(to_float(model.w)),
                   'valid': deepcopy(model.history['valid_weights'])}

        for key, weights in weights.items():
            if weights is not None:
                model.w = weights
                results[key + '_report'] = classification_report(yte, model.predict(Xte))
                results[key + '_accuracy'] = np.mean(yte == model.predict(Xte))
                results[key + '_train_accuracy'] = np.mean(ytr == model.predict(Xtr))
                c_vals = model.get_constraint_values(weights, use_float=True)
                results[key + '_constraint_values'] = c_vals
                results[key + '_max_constraint_value'] = np.max(c_vals)
                results[key + '_constraint_satisfied'] = int(np.max(c_vals) < args['constraint'])
                results[key + '_weights'] = to_float(model.w)

                # --------------------------------------------------------------------------
                # For ppercent and one protected attribute compute fractions in positive class
                if args['fairness'] == 'ppercent' and Zte.shape[1] == 1:
                    # Split into the different groups
                    for part in ['train', 'test']:
                        if part == 'train':
                            I0, I1 = (Ztr.ravel() <= 0), (Ztr.ravel() > 0)
                            X0, X1 = Xtr[I0, :], Xtr[I1, :]
                        else:
                            I0, I1 = (Zte.ravel() <= 0), (Zte.ravel() > 0)
                            X0, X1 = Xte[I0, :], Xte[I1, :]
                        n0, n1 = X0.shape[0], X1.shape[0]
                        results[key + '_z0_in_y1_' + part] = np.sum(model.predict(X0) == 1) / n0
                        results[key + '_z1_in_y1_' + part] = np.sum(model.predict(X1) == 1) / n1
                        results[key + '_z0_in_y0_' + part] = np.sum(model.predict(X0) == 0) / n0
                        results[key + '_z1_in_y0_' + part] = np.sum(model.predict(X1) == 0) / n1

                # Now print all the recorded metrics
                print("---- {} ----".format(args['optimizer']))
                print("---- {} ----".format(key))
                print('Norm Weights: ' + str(np.linalg.norm(to_float(weights))))
                print('Train Accuracy: ' + str(results[key + '_train_accuracy']))
                print('Test Accuracy: ' + str(results[key + '_accuracy']))
                if args['fairness'] == 'ppercent' and Zte.shape[1] == 1:
                    print("z0 in y1 train ", results[key + '_z0_in_y1_train'])
                    print("z1 in y1 train ", results[key + '_z1_in_y1_train'])
                    print("z0 in y1 test ", results[key + '_z0_in_y1_test'])
                    print("z1 in y1 test ", results[key + '_z1_in_y1_test'])
                print('Constraints: ' + str(results[key + '_constraint_values']))
                print('Constraint satisfied: ', results[key + '_constraint_satisfied'])
                print("\n", flush=True)

        print('Last valid epoch: ', results['last_valid_epoch'])
        print('Runtime: ' + str(results['runtime']))
        print('Lambdas: ' + str(to_float(model.lam)))
        print("final learning rate weights: ", to_float(model.lrw))
        print("final learning rate lambda: ", to_float(model.lrl))
    else:
        results['valid_accuracy'] = model.accuracy
        results['valid_train_accuracy'] = model.train_accuracy
        results['valid_z0_in_y1_train'] = model.z0_in_y1_train
        results['valid_z1_in_y1_train'] = model.z1_in_y1_train
        results['valid_z0_in_y1_test'] = model.z0_in_y1_test
        results['valid_z1_in_y1_test'] = model.z1_in_y1_test
        results['valid_constraint_satisfied'] = 1

        print('Runtime ', results['runtime'])
        print('=== Train ===')
        print('accuracy ', model.train_accuracy)
        print('y=1 in z=0 ', model.z0_in_y1_test)
        print('y=1 in z=1 ', model.z1_in_y1_test)
        print()
        print('=== Test ===')
        print('accuracy ', model.accuracy)
        print('y=1 in z=0 ', model.z0_in_y1_test)
        print('y=1 in z=1 ', model.z1_in_y1_test)

    # --------------------------------------------------------------------------
    # Export all results to pkl files
    if args['optimizer'] != 'baseline':
        directory = os.path.abspath(os.path.join(args['outdir'],
                                                 dataset,
                                                 fairness,
                                                 args['approximation'],
                                                 str(args['constraint']),
                                                 args['optimizer']))
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.join(directory, 'bs=' + str(args['batchsize']))
        if family is not None:
            filename = filename + '_bits={}.{}'.format(args['nintbits'],
                                                       args['nbits'])
    else:
        directory = os.path.abspath(os.path.join(args['outdir'],
                                                 dataset,
                                                 fairness,
                                                 str(args['constraint']),
                                                 args['optimizer']))
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.join(directory, 'bl')

    with open(filename + '.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(filename + '_args.pkl', 'wb') as handle:
        pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("DONE", flush=True)

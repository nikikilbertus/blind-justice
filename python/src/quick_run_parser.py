"""
A class to parse arguments for a single quick run.
"""

import argparse


class QuickRunParser(object):
    """
    An argument parser for all the arguments specifying a single run.

    Example:
        ```
        python quick_run.py --optimizer lagrange --dataset adult
                            --fairness ppercent --approximation secureml
                            --floats
        ```
    """

    def __init__(self):
        """Initialize a custom argument parser."""

        # Set up the parser
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
        self.parser = argparse.ArgumentParser(formatter_class=formatter_class)

        # Add command line options
        self.parser.add_argument('--outdir',
                                 help='The base directory for the output files',
                                 type=str,
                                 default='')
        self.parser.add_argument('--optimizer',
                                 help='Optimization technique',
                                 choices=['unconstrained',
                                          'lagrange',
                                          'lagrange_fixed_lambda',
                                          'projected',
                                          'iplb',
                                          'baseline'],
                                 default='lagrange')
        self.parser.add_argument('--dataset',
                                 help='One of the preset datasets',
                                 choices=['adult',
                                          'compas',
                                          'synthetic',
                                          'bank',
                                          'german',
                                          'sqf',
                                          'small_adult',
                                          'tiny_adult'],
                                 default='adult')
        self.parser.add_argument('--fairness',
                                 help='Which fairness notion to use',
                                 choices=['ppercent'],
                                 default='ppercent')
        self.parser.add_argument('--approximation',
                                 help='Approximation for sigmoid',
                                 choices=['none', 'piecewise', 'secureml'],
                                 default='secureml')
        self.parser.add_argument('--floats',
                                 help='Use floats instead of fixed point',
                                 dest='floats',
                                 action='store_true')
        self.parser.set_defaults(floats=False)
        self.parser.add_argument('--max-sample-size',
                                 help='Maximum number of training examples'
                                      'If <= 0, choose max possible.',
                                 type=int,
                                 default=-1)
        self.parser.add_argument('--nbits',
                                 help='Number of bits for the fractional part',
                                 type=int,
                                 default=16)
        self.parser.add_argument('--nintbits',
                                 help='Number of bits for the integer part',
                                 type=int,
                                 default=16)
        self.parser.add_argument('--constraint',
                                 help='The constraint value c',
                                 type=float,
                                 default=1.0)
        self.parser.add_argument('--update-rule',
                                 help='The gradient update rule',
                                 choices=['vanilla', 'momentum', 'nesterov'],
                                 default='momentum')
        self.parser.add_argument('--epochs',
                                 help='Number of epochs to run',
                                 type=int,
                                 default=10)
        self.parser.add_argument('--gradient-updates',
                                 help='Number of gradient updates, overwrites --epochs option',
                                 type=int,
                                 default=-1)
        self.parser.add_argument('--batchsize',
                                 help='Batchsize for training',
                                 type=int,
                                 default=64)
        self.parser.add_argument('--momentum',
                                 help='Momentum parameter if using momentum',
                                 type=float,
                                 default=0.95)
        self.parser.add_argument('--weight-decay',
                                 help='Weight decay parameter',
                                 type=float,
                                 default=0.0)
        self.parser.add_argument('--verbose',
                                 help='Verbosity of print output',
                                 type=int,
                                 default=0)
        self.parser.add_argument('--random-state',
                                 help='Seed for random number generation',
                                 type=int,
                                 default=123)
        self.parser.add_argument('--learning-rate',
                                 help='The learning rate to use for weights',
                                 type=float,
                                 default=1e-4)
        self.parser.add_argument('--learning-rate-lambda',
                                 help='The learning rate to use for Lagrangian multipliers',
                                 type=float,
                                 default=5e-2)
        self.parser.add_argument('--lambda-fixed',
                                 help='Fixed lambda to use for Lagrangian multipliers with fixed lambda',
                                 type=float,
                                 default=0)
        lr_decay = self.parser.add_mutually_exclusive_group(required=False)
        lr_decay.add_argument('--learning-rate-decay',
                              help='Switches on learning rate decay',
                              dest='learning_rate_decay',
                              action='store_true')
        lr_decay.add_argument('--no-learning-rate-decay',
                              help='Switches off learning rate decay',
                              dest='learning_rate_decay',
                              action='store_false')
        self.parser.set_defaults(learning_rate_decay=False)

    # -------------------------------------------------------------------------

    def parse_args(self):
        """Parse arguments and return as dict."""

        # Parse arguments and return them as a dict instead of Namespace
        return self.parser.parse_args().__dict__


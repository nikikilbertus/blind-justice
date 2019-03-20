"""
This module contains a logistic regression model that can be optimized subject
to fairness constraints.
"""

import warnings
from copy import deepcopy
import numpy as np

from tools import *


class FairLogisticRegression(object):

    def __init__(self,
                 opt=None,  # 'unconstrained', 'lagrange', 'iplb', 'projected', 'lagrange_fixed_lambda'
                 family=None,
                 constraint=1.0,
                 approx='none',
                 update_rule='vanilla',  # 'vanilla', 'momentum', 'nesterov'
                 momentum=0.95,
                 weight_decay=0.,
                 learning_rate=1e-4,
                 learning_rate_lambda=5e-2,
                 lambda_fixed=None,
                 learning_rate_decay=False,
                 t_iplb=10,
                 verbose=0,
                 random_state=42):
        """Initialize a FairLogisticRegression instance.

        Args:

            opt: The optimization method to use, either 'unconstrained', 'iplb',
                 'lagrange', 'projected', 'lagrange_fixed_lambda'
            family: The number representation (None or FXFamily)
            constraint: The constraint value
            approx: Which approximation for simgoid to use, either 'none',
                    'secureml', 'piecewise', or a callable
            update_rule: Choose from 'vanilla', 'momentum', 'nesterov'
            momentum: Momentum tu use for the optimization
            weight_decay: The weight decay factor for L2 weight decay
            learning_rate: The initial learning rate for the weights
            learning_rate_lambda: The initial learning rate for the Lagrangian
                    multipliers. Only used for opt='lagrange'
            lambda_fixed: The fixed lambda to use in opt='lagrange_fixed_lambda'
            learning_rate_decay: Learning rate decay schedule
            t_iplb: Adjusts the accuracy of the approximation of the log in
                       the interior point log barrier, lower `t_iplb` is
                       better approximation
            verbose: Verbosity of the output (int), higher is more verbose
            random_state: Seed for random numbers for reproducability
        """

        # Global parameters
        # The method used for constrained optimization
        self.opt = opt
        if self.opt is None:
            self.opt = 'unconstrained'
        # The family of the number representation (fixed point or None)
        self.family = family
        # The global constraint value
        self.raw_constraint = constraint
        self.con = to_fixed(self.raw_constraint, self.family)
        # The approximation to use for the sigmoid
        self.approx = approx
        # Verbosity of print statements
        self.verbose = verbose
        # The update rule to use
        self.update_rule = update_rule
        # The momentum for gradient updates (only for momenumt and nesterov)
        self.raw_momentum = momentum
        self.momentum = to_fixed(self.raw_momentum, self.family)
        # Weight decay term for L2 weight decay
        self.raw_weight_decay = weight_decay
        self.weight_decay = to_fixed(weight_decay, self.family)
        # Use learning rate decay or not
        self.learning_rate_decay = learning_rate_decay

        # The weights/parameters of logistic regression
        self.w = None
        # The matrix 1/n Z^T X for the constraints
        self.A = None
        # The constraint value
        self.constr_vals = None

        # Monitoring dict
        self.history = {'train_accuracy': [],
                        'test_accuracy': [],
                        'constraint_satisfied': [],
                        'constraint_values': [],
                        'valid_weights': None}

        # Specialized parameters for the lagrangian method
        self.lam = None
        self.lrw = to_fixed(learning_rate, self.family)
        self.lrl = to_fixed(learning_rate_lambda, self.family)
        self.lambda_fixed = to_fixed(lambda_fixed, self.family)

        # Specialized parameters for interior point log barrier method
        self.t_iplb = to_fixed(t_iplb, self.family)

        # Fixed point number representations
        self.one = to_fixed(np.array(1.0), self.family)
        self.zero = to_fixed(np.array(0.0), self.family)

        # What will be the batchsize (2**batch_pow)
        self.batch_pow = None

        # Will we have validation data
        self.has_val = None
        # For the learning rate decay
        self.old_val_acc = None
        self.lr_counter = None

        # Set seed
        np.random.seed(random_state)

        # Are we using an approximation of the gradient?
        if self.verbose > 0:
            # Optimization
            print("Using optimization: ", self.opt)

            # Approximation
            if self.approx is None or approx in ['None', 'none']:
                print("Using no approximation for gradients")
            elif isinstance(approx, str):
                print("Using {} for approx gradients".format(approx))
            else:
                print("Using {} for approx gradients".format(approx.__name__))

    # --------------------------------------------------------------------------
    # Different gradient routines for different optimization methods
    # --------------------------------------------------------------------------

    def loss_grad(self, X, y):
        """Compute the binary cross entropy loss and its gradient.

        Args:
            X: the features
            y: the label

        Returns:
            loss: the loss
            grad: the gradient of the loss w.r.t w
        """

        # Loss was only needed for gradient checking during debugging
        loss = None
        tmp = sub_fp(sigmoid(dot_fp(X, self.w), approx=self.approx), y)
        if self.batch_pow > 0:
            grad = div_pow2_fp(dot_fp(transpose_fp(X), tmp), self.batch_pow)
        else:
            grad = dot_fp(X, tmp)
        return loss, grad
    
    def loss_grad_lagrange(self, X, y, bce_weight=1., lam_weight=1.):
        """Compute loss and grad for constrained optimization with lagrange.

        Args:
            X: the features (batch,d)
            y: the label (batch,)
            bce_weight: weight for the bce gradient in the total gradient
            lam_weight: weight for the gradient from the constraint in the total
                gradient

        Returns:
            loss: the loss
            gradw: the gradient of the lagrangian w.r.t. w
            gradl: the gradient of the lagrangian w.r.t. lambad
        """

        loss = None

        # original binary cross entropy loss gradient
        _, grad_bce = self.loss_grad(X, y)

        # gradient of lagrange term w.r.t lambda
        inner = dot_fp(self.A, self.w)
        const = sub_fp(abs_fp(inner), self.con)
        const_not_satisfied = grt_fp(const, self.zero)
        if any_fp(const_not_satisfied):
            gradl = elem_mul_fp(const, const_not_satisfied)

            # gradient of lagrange term w.r.t w
            inner_pos = grt_fp(inner, self.zero)
            inner_neg = grt_fp(self.zero, inner)
            sign = sub_fp(elem_mul_fp(inner_pos, const_not_satisfied),
                          elem_mul_fp(inner_neg, const_not_satisfied))
            grad_const = dot_fp(transpose_fp(self.A),
                                elem_mul_fp(self.lam, sign))
        else:
            gradl = to_fixed(np.zeros(len(self.lam)), family=self.family)
            grad_const = to_fixed(np.zeros(len(self.w)), family=self.family)

        gradw = add_fp(dot_fp(bce_weight, grad_bce),
                       dot_fp(lam_weight, grad_const))
        return loss, gradw, gradl

    def loss_grad_lagrange_fixed_lambda(self, X, y):
        """Compute loss and grad for constrained optimization with Lagrangian
           multipliers, but fixed lambda.

        Args:
            X: the features (batch,d)
            y: the label (batch,)

        Returns:
            loss: the loss
            gradw: the gradient of the lagrangian w.r.t. w
        """

        loss = None

        # original binary cross entropy loss gradient
        _, grad_bce = self.loss_grad(X, y)

        inner = dot_fp(self.A, self.w)
        const = sub_fp(abs_fp(inner), self.con)
        const_not_satisfied = grt_fp(const, self.zero)
        if any_fp(const_not_satisfied):
            # gradient of lagrange term w.r.t w
            inner_pos = grt_fp(inner, self.zero)
            inner_neg = grt_fp(self.zero, inner)
            sign = sub_fp(elem_mul_fp(inner_pos, const_not_satisfied),
                          elem_mul_fp(inner_neg, const_not_satisfied))
            grad_const = dot_fp(transpose_fp(self.A),
                                elem_mul_fp(self.lam, sign))
        else:
            grad_const = to_fixed(np.zeros(len(self.w)), family=self.family)

        gradw = add_fp(grad_bce, grad_const)
        return loss, gradw

    def loss_grad_iplb(self, X, y):
        """Compute loss and grad for the interior point log barrier method.

        Args:
            X: the features (batch,d)
            y: the label (batch,)

        Returns:
            loss: the loss
            grad: the gradient of the loss w.r.t w
        """
        loss = None

        # gradient from original binary cross entropy loss
        _, grad_bce = self.loss_grad(X, y)

        # gradient from log barrier
        inner = dot_fp(self.A, self.w)
        tmp1 = sub_fp(self.con, inner)
        tmp2 = add_fp(self.con, inner)
        # Add some epsilon before division for stability
        grad_con = dot_fp(transpose_fp(self.A),
                          sub_fp(inv_fp(add_fp(tmp1, 1e-4)),
                                 inv_fp(add_fp(tmp2, 1e-4))))
        return loss, add_fp(grad_bce, elem_div_fp(grad_con, self.t_iplb))

    def loss_grad_projected(self, X, y):
        """Compute loss and grad for the projected gradient method.

        Args:
            X: the features (batch,d)
            y: the label (batch,)

        Returns:
            loss: the loss
            grad: the gradient of the loss w.r.t w
        """
        loss = None

        # gradient from original binary cross entropy loss
        _, grad = self.loss_grad(X, y)

        # get the active constraints
        constr_vals = self.get_constraint_values(self.w)
        active = [i for i, c in enumerate(constr_vals) if c >= self.con]
        # project active constraints back into constraint set
        if not active:
            return loss, grad
        else:
            return loss, project_to(grad, self.A, active)

    # --------------------------------------------------------------------------
    # The main fit function
    # --------------------------------------------------------------------------

    def fit(self, X, y, Z, epochs=50, batchsize=64, Xval=None, yval=None):
        """Train logistic regression subject to the fairness constraints.

        Args:
            X: numpy.ndarray of features (n_samples, n_features)
            y: numpy.ndarray of labels (n_samples,)
            Z: numpy.ndarray of protected attr. (n_samples, n_protected)
            epochs: How many epochs to use for training
            batchsize: The batch size to use
            Xval: optional validation features (n_val_samples, n_features)
            yval: optional validation labels (n_val_samples, n_features)
        """

        # Are we using floating point or fixed point arithmetic
        if self.verbose > 0:
            if self.family is None:
                print("Treat data as {}".format(type(X)))
            else:
                print("Treat data as {}".format(self.family))

        # Do we have validation data
        self.has_val = Xval is not None and yval is not None

        # Set old validation accuracy and counter for learning rate decay
        if self.has_val and self.learning_rate_decay:
            self.old_val_acc = to_fixed(0.0, self.family)
            self.lr_counter = to_fixed(0, self.family)

        # Make sure batchsize is power of two and get exponent for fast div
        self.batch_pow = get_pow2(batchsize)
        assert batchsize == 2**self.batch_pow, "Batchsize must be power of 2."

        # Convert data to fixed point if necessary
        X = to_fixed(X, self.family)
        y = to_fixed(y, self.family)
        Z = to_fixed(Z, self.family)

        # Some relevant dimensions of the problem
        n = dimensions(X)[0]
        d = dimensions(X)[1]
        p = dimensions(Z)[1]

        # Initialize weights and placeholder variable for weight updates
        self.w = to_fixed(np.zeros((d,)), self.family)
        # Only for momentum or nesterov
        v = to_fixed(np.zeros((d,)), self.family)
        # Only for lagrange with momentum or nesterov
        vw = to_fixed(np.zeros((d,)), self.family)
        vl = to_fixed(np.zeros((p,)), self.family)

        # Only for lagrange optimization
        if self.opt == 'lagrange':
            self.lam = to_fixed(np.zeros(p), self.family)
        elif self.opt == 'lagrange_fixed_lambda':
            self.lam = to_fixed(np.ones(p) * to_float(self.lambda_fixed),
                                self.family)

        # Check whether initial weights satisfy constraints
        self.A = blocked_mat_mat_dot_avg_fp(transpose_fp(Z), X)
        if not self._check_constraint(self.w):
            warnings.warn("Constraints initially not satisfied, c={}"
                          .format(self.con))
        elif self.verbose > 0:
            print("\nConstraints are satisfied for initial parameters.\n")

        # Training loop
        for t in range(epochs):
            if self.verbose > 0:
                print('\n epoch: ' + str(t))

            # Shuffle data for each epoch
            perm = np.random.permutation(n)
            if self.family is None:
                Xp, yp, Zp = X[perm], y[perm], Z[perm]
            else:
                Xp = [X[i] for i in perm]
                yp = [y[i] for i in perm]
                Zp = [Z[i] for i in perm]

            # Loop over the minibatches
            for i1 in range(0, n, batchsize):
                # Extract a minibatch
                i2 = min(i1 + batchsize, n)
                Xb, yb, Zb = Xp[i1:i2], yp[i1:i2], Zp[i1:i2]

                # Now we check for the optimization routine to call
                # Ignore constraints
                if self.opt == 'unconstrained':
                    loss, grad = self.loss_grad(Xb, yb)
                    self.w, v = self._update_step(self.lrw, grad, self.w, v)

                # Constrained optimization using Lagrangian multipliers
                elif self.opt == 'lagrange':
                    bce_weight = epochs / (epochs + t)
                    lam_weight = (epochs + 10 * t) / epochs
                    _, gradw, gradl = self.loss_grad_lagrange(Xb, yb,
                                                              bce_weight=bce_weight,
                                                              lam_weight=lam_weight)
                    self.w, vw = self._update_step(self.lrw, gradw, self.w, vw)
                    self.lam, vl = self._update_step(self.lrl, neg_fp(gradl),
                                                     self.lam, vl)
                    lambda_pos = grt_fp(self.lam, self.zero)
                    self.lam = elem_mul_fp(self.lam, lambda_pos)

                # Constrained opt using Lagrangian multipliers with fixed lambda
                elif self.opt == 'lagrange_fixed_lambda':
                    _, gradw = self.loss_grad_lagrange_fixed_lambda(Xb, yb)
                    self.w, vw = self._update_step(self.lrw, gradw, self.w, vw)

                # Constrained optimization with an interior point log barrier
                elif self.opt == 'iplb':
                    _, grad = self.loss_grad_iplb(Xb, yb)
                    self.w, v = self._update_step(self.lrw, grad, self.w, v)

                # projected gradient
                elif self.opt == 'projected':
                    _, grad = self.loss_grad_projected(Xb, yb)
                    self.w, v = self._update_step(self.lrw, grad, self.w, v)

                # Unknown optimization method
                else:
                    raise Exception('Unknown optimization {}'.format(self.opt))

            # Learning rate decay
            # NEEDS FLOAT COMPUTATION if validation set available
            if self.learning_rate_decay:
                self._decay_learning_rate(Xval, yval, t)

            # Increase parameter t in iplb for better approximation of step
            if self.opt == 'iplb':
                self.t_iplb *= 10.**(1. / epochs)
                if (2*p) / self.t_iplb < 1e-4:
                    warnings.warn("Achieved t/m smaller than 1e-4 in iplb")

            # Checkpoint training and test accuracy
            train_acc = self.accuracy(X, y)
            if self.has_val:
                test_acc = self.accuracy(Xval, yval)
            else:
                test_acc = -1
            self.history['train_accuracy'].append(train_acc)
            self.history['test_accuracy'].append(test_acc)

            # Set the constraint values for the current weights
            self.constr_vals = self.get_constraint_values(self.w)

            # Checkpoint constraint values, satisfaction and weights
            c_vals = to_float(self.constr_vals)
            sat = int(np.max(c_vals) < self.con)
            self.history['constraint_values'].append(c_vals)
            self.history['constraint_satisfied'].append(sat)
            if sat:
                self.history['valid_weights'] = to_float(self.w)

            # Print monitoring statements
            if self.verbose > 0:
                print("train accuracy: ", train_acc)
                print("test accuracy: ", test_acc)
                print('\n', flush=True)

        if self.opt == 'lagrange':
            self._fix_constraint(X, y, vw, vl, Xval, yval)

        # Check whether final weights satisfy constraints and warn otherwise
        if not self._check_constraint(self.w):
            warnings.warn("Constraints finally not satisfied, c={}, opt={}"
                          .format(self.con, self.opt), UserWarning)
        elif self.verbose > 0:
            print("\nConstraints are satisfied for final parameters.\n")

    # --------------------------------------------------------------------------
    # Predict on new data
    # --------------------------------------------------------------------------

    def predict_proba(self, X):
        """Use trained weights to predict probabilities."""
        assert self.w is not None, "No trained parameters."
        return sigmoid(np.dot(to_float(X), to_float(self.w)))

    def predict(self, X):
        """Use trained weights to predict binary outcomes."""
        yhat = np.round(self.predict_proba(X))
        return yhat

    def accuracy(self, X, y):
        """Compute the accuracy using floats"""
        return np.mean(to_float(y) == self.predict(X))

    # --------------------------------------------------------------------------
    # Helper functions of various kinds
    # --------------------------------------------------------------------------

    def _fix_constraint(self, X, y, vw, vl, Xval, yval):
        extra_rounds = 0
        while not self._check_constraint(self.w) and extra_rounds < 1000:
            bce_weight = 0
            lam_weight = 1
            _, gradw, gradl = self.loss_grad_lagrange(X, y,
                                                      bce_weight=bce_weight,
                                                      lam_weight=lam_weight)
            self.w, vw = self._update_step(self.lrw, gradw, self.w, vw)
            self.lam, vl = self._update_step(self.lrl, neg_fp(gradl),
                                             self.lam, vl)
            lambda_pos = grt_fp(self.lam, self.zero)
            self.lam = elem_mul_fp(self.lam, lambda_pos)
            extra_rounds += 1
            print("Extra rounds: ", extra_rounds)

        # Checkpoint training and test accuracy
        train_acc = self.accuracy(X, y)
        if self.has_val:
            test_acc = self.accuracy(Xval, yval)
        else:
            test_acc = -1
        self.history['train_accuracy'].append(train_acc)
        self.history['test_accuracy'].append(test_acc)

        # Set the constraint values for the current weights
        self.constr_vals = self.get_constraint_values(self.w)

        # Checkpoint constraint values and satisfaction
        c_vals = to_float(self.constr_vals)
        sat = int(np.max(c_vals) <= self.con)
        self.history['constraint_values'].append(c_vals)
        self.history['constraint_satisfied'].append(sat)
        if sat:
            self.history['valid_weights'] = to_float(self.w)

    def _decay_learning_rate(self, Xval, yval, epoch):
        """If learning rate needs decay, try one decay step.

            If we have access to a validation set, we decay the learning rate(s)
            by a constant factor of 2 every 3 epochs in which the validation
            accuracy has not improved.

            If we don't have access to a validation set, we decay the learning
            rate(s) by a constant factor of 2 every 5 epochs.

            WARNING: These values heuristically chosen and barely optimized.
        """
        decay = False
        # Check if we need to decay this epoch
        if self.has_val:
            val_acc = np.mean(to_float(yval) == self.predict(Xval))
            if val_acc < self.old_val_acc:
                self.lr_counter += 1
                if self.lr_counter > 10:
                    decay = True
                    self.lr_counter = 0
                    self.old_val_acc = val_acc
            else:
                self.old_val_acc = val_acc
                self.lr_counter = 0
        else:
            if epoch % 5 == 0:
                decay = True
        # Actually decay the learning rate
        if decay:
            self.lrw = maximum_fp(1e-6, self.lrw * div_pow2_fp(self.lrw, 1))
            self.lrl = maximum_fp(1e-6, self.lrl * div_pow2_fp(self.lrl, 1))

    def _update_step(self, lr, grad, w, v):
        """Perform one update step of the weights."""
        if self.weight_decay == 0:
            wtmp = w
        else:
            wtmp = dot_fp(1. - self.weight_decay, w)

        # Implementation: Vanilla SGD
        if self.update_rule == 'vanilla':
            wnew = sub_fp(wtmp, dot_fp(lr, grad))

        # Momentum update
        # Plain description:
        # v = mu * v - learning_rate * dx # integrate velocity
        # x += v # integrate position
        elif self.update_rule == 'momentum':
            v = sub_fp(dot_fp(self.momentum, v), dot_fp(lr, grad))
            wnew = add_fp(wtmp, v)

        # SGD with Nesterov Momentum
        # Plain description:
        # v_prev = v # back this up
        # v = mu * v - learning_rate * dx # velocity update stays the same
        # x += -mu * v_prev + (1 + mu) * v # position update changes form
        elif self.update_rule == 'nesterov':
            v_prev = deepcopy(v)
            v = sub_fp(dot_fp(self.momentum, v), dot_fp(lr, grad))
            wnew = add_fp(wtmp,
                          add_fp(dot_fp(-self.momentum, v_prev),
                                 dot_fp(1. + self.momentum, v)))
        else:
            raise ValueError("Unknown update rule {}".format(self.update_rule))

        return wnew, v

    def _check_constraint(self, w, use_float=False):
        """Check whether the constraints are satisfied.

        Args:
            w: Weight parameters to check
            use_float: bool, use floats for fast computation?

        Returns:
            bool whether constraints are satisfied or not
        """
        vals = self.get_constraint_values(w, use_float=use_float)

        if use_float:
            c = to_float(self.con)
        else:
            c = self.con

        for x in vals:
            if x > c:
                return False
        return True

    def get_constraint_values(self, w, use_float=False):
        """Get the constraint values

        Args:
            w: the weights
            use_float: bool, use floats for fast computation?

        Returns:
            an iterable of the constraint values
        """
        if use_float:
            return get_iterable(abs_fp(dot_fp(to_float(self.A), to_float(w))))
        else:
            return get_iterable(abs_fp(dot_fp(self.A, w)))

    def _print_metrics(self, X, y, Xval, yval):
        """Print some training metrics.

        Args:
            X: The features
            y: The labels
            Xval:  The validation features
            yval:  The validation labels

        Returns:
            training accuracy, test accuracy
        """
        # Can be done with fast numpy routines
        inner = sigmoid(to_float(X).dot(to_float(self.w)))
        loss = -np.sum(to_float(y) * np.log(inner) +
                       (1. - to_float(y)) * np.log(1. - inner))

        # Loss computation with fixed point arithmetic
        print('train loss: ' + str(loss))

        # Training set accuracy
        train_acc = self.accuracy(X, y)
        print("training accuracy: ", train_acc)

        # Validation set accuracy if provided
        if self.has_val:
            test_acc = self.accuracy(Xval, yval)
            print("validation accuracy: ", test_acc)

        print("\n", flush=True)


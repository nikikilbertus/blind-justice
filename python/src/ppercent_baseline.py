"""
This module contains the baseline for the p percent rule from Zafar et al.:
https://github.com/mbilalzafar/fair-classification
"""

import numpy as np
import baseline_utils as ut
from data import _labels_to_zero_one, _labels_to_plus_minus


class PpercentBaseline(object):
    """
    This is a wrapper class around code from Zafar et al.

    We use this as a baseline to compare against.
    For more information and the original code see
    https://github.com/mbilalzafar/fair-classification.
    """

    def __init__(self, constraint=1.):
        """Initialize an instance of PpercentBaseline."""
        self.constraint = constraint
        self.w = None
        self.accuracy = None
        self.train_accuracy = None
        self.z0_in_y1_test = None
        self.z1_in_y1_test = None
        self.z0_in_y0_test = None
        self.z1_in_y0_test = None
        self.z0_in_y1_train = None
        self.z1_in_y1_train = None
        self.z0_in_y0_train = None
        self.z1_in_y0_train = None

    def fit(self, X, y, Z, Xval=None, yval=None, Zval=None):
        """
        Fit the constrained logistic regression problem

        Note that we feed the data in the same form as to the other models.
        However, we have to then convert it to use 0 and 1 labels and map
        the centered Z back to 0 1 arrays.

        Args:
            X: training features
            y: training labels
            Z: training sensitive attributes
            Xval: test features
            yval: test labels
            Zval: test sensitive attributes

        Returns:
            None
        """

        # Use plus minus 1 as labels
        _labels_to_plus_minus(y, yval)
        # Get the input data into the correct format
        p = Z.shape[1]
        Zdict = {}
        Zvaldict = {}
        for i in range(p):
            z = Z[:, i]
            zval = Zval[:, i]
            # Map the sensitive attributes back to being 0 or 1
            _labels_to_zero_one(z, zval)
            Zdict['s' + str(i)] = z
            Zvaldict['s' + str(i)] = zval
            ut.compute_p_rule(z, y)  # compute the p-rule in the original data
        Z = Zdict
        Zval = Zvaldict

        loss_function = self._logistic_loss
        sensitive_attrs = ['s' + str(i) for i in range(p)]

        # optimize accuracy subject to fairness constraints
        apply_fairness_constraints = 1
        sensitive_attrs_to_cov_thresh = {'s' + str(i): self.constraint
                                         for i in range(p)}
        print()
        print("== Classifier with fairness constraint ==")

        w = ut.train_model(X, y, Z, loss_function, apply_fairness_constraints,
                           sensitive_attrs, sensitive_attrs_to_cov_thresh)
        train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(w, X, y,
                                                                                                 Xval, yval, None,
                                                                                                 None)
        distances_boundary_test = (np.dot(Xval, w)).tolist()
        all_class_labels_assigned_test = np.sign(distances_boundary_test)
        correlation_dict_test = ut.get_correlations(None, None, all_class_labels_assigned_test, Zval,
                                                    sensitive_attrs)
        cov_dict_test = ut.print_covariance_sensitive_attrs(None, Xval, distances_boundary_test, Zval,
                                                            sensitive_attrs)
        p_rule = ut.print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test],
                                                    sensitive_attrs[0])
        self.w = w
        self.accuracy = self._accuracy(Xval, yval)
        self.train_accuracy = self._accuracy(X, y)

        z0y1, z1y1, z0y0, z1y0 = self._get_fractions_in_groups(X, Z)
        self.z0_in_y1_train = z0y1
        self.z1_in_y1_train = z1y1
        self.z0_in_y0_train = z0y0
        self.z1_in_y0_train = z1y0

        z0y1, z1y1, z0y0, z1y0 = self._get_fractions_in_groups(Xval, Zval)
        self.z0_in_y1_test = z0y1
        self.z1_in_y1_test = z1y1
        self.z0_in_y0_test = z0y0
        self.z1_in_y0_test = z1y0

    def _predict_proba(self, X):
        """Use trained weights to predict probabilities."""
        assert self.w is not None, "No trained parameters."
        inner = np.dot(X, self.w)
        return 1. / (1. + np.exp(-inner))

    def _predict(self, X):
        """Use trained weights to predict binary outcomes."""
        yhat = np.round(self._predict_proba(X))
        _labels_to_plus_minus(yhat)
        return yhat

    def _accuracy(self, X, y):
        """Compute the accuracy using floats"""
        return np.mean(y == self._predict(X))

    def _get_fractions_in_groups(self, X, Z, sensitive_index=1):
        if len(Z) == 1:
            I0, I1 = (Z['s0'] <= 0), (Z['s0'] > 0)
        else:
            I0, I1 = (Z['s' + str(sensitive_index)].ravel() <= 0),\
                     (Z['s' + str(sensitive_index)].ravel() > 0)
        X0, X1 = X[I0, :], X[I1, :]
        n0, n1 = X0.shape[0], X1.shape[0]
        yhat0 = self._predict(X0)
        yhat1 = self._predict(X1)
        z0_in_y1 = np.sum(yhat0 > 0) / n0
        z1_in_y1 = np.sum(yhat1 > 0) / n1
        z0_in_y0 = np.sum(yhat0 <= 0) / n0
        z1_in_y0 = np.sum(yhat1 <= 0) / n1
        return z0_in_y1, z1_in_y1, z0_in_y0, z1_in_y0

    def _logistic_loss(self, w, X, y, return_arr=None):
        """Computes the logistic loss.

        This function is used from scikit-learn source code

        Parameters
        ----------
        w : ndarray, shape (n_features,) or (n_features + 1,)
            Coefficient vector.

        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.

        y : ndarray, shape (n_samples,)
            Array of labels.

        """
        yz = y * np.dot(X, w)
        # Logistic loss is the negative of the log of the logistic function.
        if return_arr == True:
            out = -(self._log_logistic(yz))
        else:
            out = -np.sum(self._log_logistic(yz))
        return out

    def _log_logistic(self, X):
        """ This function is used from scikit-learn source code. Source link below """

        """Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.
        This implementation is numerically stable because it splits positive and
        negative values::
            -log(1 + exp(-x_i))     if x_i > 0
            x_i - log(1 + exp(x_i)) if x_i <= 0

        Parameters
        ----------
        X: array-like, shape (M, N)
            Argument to the logistic function

        Returns
        -------
        out: array, shape (M, N)
            Log of the logistic function evaluated at every point in x
        Notes
        -----
        Source code at:
        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
        -----

        See the blog post describing this implementation:
        http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
        """
        if X.ndim > 1:
            raise Exception("Array of samples cannot be more than 1-D!")
        out = np.empty_like(X)  # same dimensions and data types

        idx = X > 0
        out[idx] = -np.log(1.0 + np.exp(-X[idx]))
        out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
        return out


if __name__ == "__main__":
    import data
    xtr, xte, ytr, yte, ztr, zte = data.read_syntehtic()
    pp = PpercentBaseline(constraint=0.0001)
    pp.fit(xtr, ytr, ztr, xte, yte, zte)

    print('=== Train ===')
    print('accuracy ', pp.train_accuracy)
    print('y=1 in z=0 ', pp.z0_in_y1_test)
    print('y=1 in z=1 ', pp.z1_in_y1_test)
    print()
    print('=== Test ===')
    print('accuracy ', pp.accuracy)
    print('y=1 in z=0 ', pp.z0_in_y1_test)
    print('y=1 in z=1 ', pp.z1_in_y1_test)

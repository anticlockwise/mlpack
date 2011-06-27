import numpy
from mlpack.svm.kernel import *

CLASSIFICATION = 1
REGRESSION     = 2

class SvmParameters:
    def __init__(self):
        self.svm_type              = CLASSIFICATION
        self.c                     = 0.0
        self.eps                   = 0.1
        self.costratio             = 1.0
        self.costratio_unlab       = 1.0
        self.unlabbound            = 0.00001
        self.transduction_posratio = -1.0
        self.biased_hyperplane     = 1
        self.sharedslack           = 0
        self.remove_inconsistent   = 0
        self.skip_final_opt_check  = 0
        self.maxqpsize             = 10
        self.newvarsinqp           = 0
        self.iter_to_shrink        = -9999
        self.maxiter               = 10000
        self.kernel_cache_size     = 40
        self.eps_crit              = 0.001
        self.eps_a                 = 10**-15
        self.compute_loo           = 0
        self.rho                   = 1.0
        self.xa_depth              = 0
        self.num_preds             = 0
        self.cost                  = None

class SvmModel:
    def __init__(self, num_docs=0, num_preds=0):
        self.num_sv         = 1
        self.at_upper_bound = 0
        self.b              = 0.0
        self.supvec         = None
        self.alpha          = None
        self.index          = None
        self.num_preds      = num_preds
        self.num_docs       = num_docs

        self.loo_error      = 1.0
        self.loo_recall     = 1.0
        self.loo_precision  = 1.0
        self.xa_error       = 1.0
        self.xa_recall      = 1.0
        self.xa_precision   = 1.0
        self.lin_weights    = None

        self.maxdiff        = 0.0

    def _adjust_float(self, f):
        if f == -1.0:
            return -1.0000002
        return f

    def __str__(self):
        self.loo_error = self._adjust_float(self.loo_error)
        self.loo_recall = self._adjust_float(self.loo_recall)
        self.loo_precision = self._adjust_float(self.loo_precision)
        self.xa_error = self._adjust_float(self.xa_error)
        self.xa_recall = self._adjust_float(self.xa_recall)
        self.xa_precision = self._adjust_float(self.xa_precision)
        self.b = self._adjust_float(self.b)
        self.maxdiff = self._adjust_float(self.maxdiff)
        s = """
Number of support vectors: %d
At upper bound:            %d
Number of predicates:      %d
Number of documents:       %d
Biased hyperplane:         %.2f

LOO Error:                 %+.2f
LOO Recall:                %+.2f
LOO Precision:             %+.2f
XA Error:                  %+.2f
XA Recall:                 %+.2f
XA Precision:              %+.2f

Max Diff:                  %+.2f

Alphas:                    %s
Indexes:                   %s
Linear weights:            %s
""" % (self.num_sv, self.at_upper_bound, self.num_preds, self.num_docs, \
        self.b, self.loo_error, self.loo_recall, self.loo_precision, \
        self.xa_error, self.xa_recall, self.xa_precision, self.maxdiff, \
        self.alpha, self.index, self.lin_weights)
        return s

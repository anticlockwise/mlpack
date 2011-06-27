import numpy
from mlpack.svm.kernel import *

CLASSIFICATION = 0
REGRESSION     = 1

class SVector(object):
    def __init__(self, num_preds=1, kernel_id=0, factor=1.0):
        self.preds      = numpy.zeros((num_preds,))
        self.kernel_id  = kernel_id
        self.factor     = factor
        self.twonorm_sq = 0.0
        self.attrs      = {}

    def __mul__(self, other):
        return numpy.dot(self.preds, other.preds)[0]

class Document(object):
    def __init__(self, idx=-2, y=-1, qid=0, sid=0,
            cost_factor=0.0):
        self.idx         = idx
        self.preds       = SVector()
        self.y           = y
        self.qid         = qid
        self.sid         = sid
        self.cost_factor = cost_factor

    def num_preds(self):
        return len(self.preds.preds)

class SvmParameters(object):
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
        self.maxiter               = 100000
        self.kernel_cache_size     = 40
        self.eps_crit              = 0.001
        self.eps_a                 = 10**-15
        self.compute_loo           = 0
        self.rho                   = 1.0
        self.xa_depth              = 0
        self.num_preds             = 0
        self.cost                  = None

class SvmModel(object):
    def __init__(self, num_docs=0, num_preds=0, kernel_config=None):
        self.num_sv         = 1
        self.at_upper_bound = 0
        self.b              = 0.0
        self.supvec         = [None for i in range(num_docs+2)]
        self.alpha          = numpy.zeros((num_docs+2,))
        self.index          = numpy.zeros((num_docs+2,))
        self.num_preds      = num_preds
        self.num_docs       = num_docs
        self.kernel_config  = kernel_config

        self.loo_error      = -1.0
        self.loo_recall     = -1.0
        self.loo_precision  = -1.0
        self.xa_error       = -1.0
        self.xa_recall      = -1.0
        self.xa_precision   = -1.0
        self.lin_weights    = None

        self.maxdiff        = 0.0

class ShrinkState(object):
    def __init__(self):
        self.deactnum       = 0
        self.active         = None
        self.inactive_since = None
        self.a_history      = None
        self.max_history    = 0
        self.last_a         = None
        self.last_lin       = None

from cStringIO import StringIO

class MaxentParameters(object):
    """
    Parameters to be used in Maxent model learning.
    """
    def __init__(self, params, corr_param, corr_constant,
            n_outcomes):
        """
        Initialize a parameters object

        :type  params:     list
        :param params:     The list of parameters corresponding to the predicates
        :type  corr_param: number
        :param corr_param: Correction parameter
        :type  corr_constant: number
        :param corr_constant: Correct constant for features
        :type  n_outcomes: number
        :param n_outcomes: The number of outcomes for this Maxent model
        """
        self.params = params
        self.corr_param = corr_param
        self.corr_constant = corr_constant
        self.n_outcomes = n_outcomes
        self.const_inverse = 1.0 / corr_constant

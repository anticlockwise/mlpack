from math import exp
from mlpack.model import *
from mlpack.maxent.params import *
from cStringIO import StringIO
import numpy

class MaxentModel(Model):
    """
    Class to store parameters for a trained Maximum Entropy Model
    """

    def __init__(self, params, plabels, olabels, prior):
        """
        Initialize this model with the parameters, outcome labels, predicate labels
        and prior distribution information

        :type  params:  list
        :param params:  List of parameters for the retained predicates - p(x, y)
        :type  plabels: list
        :param plabels: String representations of the predicates
        :type  olabels: list
        :param olabels: String representations of the outcomes
        :type  prior:   :py:mod:`mlpack.prior`
        :param prior:   Prior distribution
        """
        Model.__init__(self, olabels, plabels)
        self.prior = prior
        self.params = MaxentParameters(params, 0.0, 1.0, len(olabels))

    def eval(self, context):
        """
        Evaluate the probabilities of producing each outcome for the predicate context,
        given this model.

        :type  context: :py:class:`mlpack.events.FeatureSet`
        :param context: Predicates context to evaluate

        :rtype:         list
        :return:        The list of probabilities for each different outcome.
        """
        dist = self.prior.log_prior(context)
        return self.evaluate(context, dist, self.params)

    @classmethod
    def evaluate(cls, context, prior, model):
        """
        Class method for evaluating the probabilities of producing each outcome
        for the given context based on the given prior and model parameters.

        :type  context: :py:class:`mlpack.events.FeatureSet`
        :param context: Predicates context to evaluate
        :type  prior:   :py:mod:`mlpack.prior`
        :param prior:   The prior distribution to use
        :type  model:   list
        :param model:   The list of parameters for the predicates - p(x, y)

        :rtype:         list
        :return:        The list of probabilities for each different outcome.
        """
        params = model.params
        n_feats = numpy.zeros((model.n_outcomes,))
        active_outcomes = []
        active_params = []
        value = 1.0
        for feature in context:
            if feature.index != -1:
                p = params[feature.index]
                active_outcomes = p.outcomes
                active_params = p.params
                value = feature.value
                for ai, oid in enumerate(active_outcomes):
                    n_feats[oid] += 1
                    prior[oid] += active_params[ai] * value

        normal = 0.0
        for oid in range(model.n_outcomes):
            if model.corr_param != 0:
                prior[oid] = exp(prior[oid] * model.const_inverse\
                        + ((1.0 - (n_feats[oid] / model.corr_constant))\
                        * model.corr_param))
            else:
                prior[oid] = exp(prior[oid] * model.const_inverse)

            normal += prior[oid]

        for oid in range(model.n_outcomes):
            prior[oid] /= normal

        return prior

    def __str__(self):
        sio = StringIO()
        sio.write("\n")
        sio.write("Number of predicates: %d\n" % len(self.plabels))
        sio.write("  Number of outcomes: %d\n" % len(self.olabels))
        sio.write("\n")
        sio.write("Parameters:\n")
        s = sio.getvalue()
        for pi, param in enumerate(self.params.params):
            sio.write("%s: " % self.plabels[pi])
            ps = "\t".join("%s[%.2f]" % (self.olabels[oi], param.params[aoi])\
                    for aoi,oi in enumerate(param.outcomes))
            sio.write(ps)
            sio.write("\n")
        s = sio.getvalue()
        return s

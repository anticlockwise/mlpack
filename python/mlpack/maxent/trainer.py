from mlpack.trainer import Trainer
from mlpack.maxent.model import *
from mlpack.maxent.params import *
from mlpack.params import *
from mlpack.prior import *
from math import ceil, log
import numpy
import sys

class GISTrainer(Trainer):
    """
    GIS training algorithm implementation.
    """
    def __init__(self):
        self.use_simple_smoothing   = False
        self.use_slack_param        = False
        self.use_gaussian_smoothing = False
        self.sigma                  = 2.0
        self.smoothing_observation  = 0.1
        self.tolerance              = 0.0001
        self.prior                  = None

    def train(self, di, config):
        """
        Main training algorithm function.

        :type  di: :py:class:`mlpack.index.DataIndexer`
        :param di: DataIndexer object that includes indexed training data.
        :type  config: ConfigParser.ConfigParser
        :param config: Configuration for the training algorithm.
        """
        iterations                  = config.getint("maxent", "iterations")
        prior_type                  = config.get("maxent", "gis_prior")
        self.cutoff                 = config.getint("maxent", "cutoff")
        self.use_simple_smoothing   = config.getboolean("maxent", "gis_simple_smoothing")
        self.smoothing_observation  = config.getfloat("maxent", "gis_smoothing_observation")
        self.use_gaussian_smoothing = config.getboolean("maxent", "gis_gaussian_smoothing")
        self.use_slack_param        = config.getboolean("maxent", "gis_slack_param")
        self.sigma                  = config.getfloat("maxent", "gis_sigma")
        self.tolerance              = config.getfloat("maxent", "gis_tolerance")

        if prior_type == "uniform":
            self.prior = UniformPrior()

        print "Incorporating indexed data for training..."
        self.contexts = di.contexts()
        self.pred_counts = di.pred_counts()
        self.n_uniq_events = len(self.contexts)

        corr_constant = 1
        for event in self.contexts:
            context = event.context
            cl = reduce(lambda f, y: y + f, context).value
            if cl > corr_constant:
                corr_constant = ceil(cl)

        print "done."

        self.outcome_labels = di.outcome_labels()
        self.pred_labels = di.pred_labels()
        self.n_outcomes = len(self.outcome_labels)
        self.n_preds = len(self.pred_labels)

        self.prior.set_labels(self.outcome_labels, self.pred_labels)

        print "\tNumber of predicates: %d" % self.n_preds
        print "\t    Number of events: %d" % self.n_uniq_events
        print "\t  Number of outcomes: %d" % self.n_outcomes

        pred_count = numpy.zeros((self.n_preds, self.n_outcomes))
        for event in self.contexts:
            context = event.context
            for feature in context:
                pred_count[feature.index][event.oid] += event.count * feature.value

        self.params = [Parameters() for i in range(self.n_preds)]
        self.model_expects = [Parameters() for i in range(self.n_preds)]
        self.observed_expects = [Parameters() for i in range(self.n_preds)]

        outcome_pattern = []
        all_outcomes = [i for i in range(self.n_outcomes)]
        active_outcomes = [0 for i in range(self.n_outcomes)]
        n_active_outcomes = 0
        for pi in range(self.n_preds):
            n_active_outcomes = 0
            if self.use_simple_smoothing:
                n_active_outcomes = self.n_outcomes;
                outcome_pattern = all_outcomes
            else:
                for oi in range(self.n_outcomes):
                    if pred_count[pi][oi] > 0 and self.pred_counts[pi] >= self.cutoff:
                        active_outcomes[n_active_outcomes] = oi
                        n_active_outcomes += 1
                outcome_pattern = active_outcomes[0:n_active_outcomes]

            self._update(self.params[pi], outcome_pattern, n_active_outcomes)
            self._update(self.model_expects[pi], outcome_pattern, n_active_outcomes)
            self._update(self.observed_expects[pi], outcome_pattern, n_active_outcomes)

            for aoi in range(n_active_outcomes):
                oi = outcome_pattern[aoi]
                self.params[pi].set(aoi, 0.0)
                self.model_expects[pi].set(aoi, 0.0)
                if pred_count[pi][oi] > 0:
                    self.observed_expects[pi].set(aoi, pred_count[pi][oi])
                elif use_simple_smoothing:
                    self.observed_expects[pi].set(aoi, smoothing_observation)

        if self.use_slack_param:
            pass

        self.num_feats = [0 for i in range(self.n_outcomes)]

        print "...done."
        self.eval_params = MaxentParameters(self.params, 0.0, 1.0, self.n_outcomes)
        print "Computing model parameters..."
        self._find_params(iterations, corr_constant)

        model = MaxentModel(self.params, self.pred_labels, self.outcome_labels, self.prior)

        print model

        return model

    def set_heldout_data(self, events):
        """
        """
        pass

    def _update(self, params, outcomes, n_active_outcomes):
        params.outcomes = outcomes
        params.params = numpy.zeros((n_active_outcomes,))

    def _find_params(self, iterations, corr_constant):
        prev_ll, curr_ll = 0.0, 0.0

        print "Performing %d iterations" % iterations
        for i in range(iterations):
            sys.stdout.write("%d:  " % (i+1,))
            curr_ll = self._next_iteration(corr_constant)
            if i > 0:
                if prev_ll > curr_ll:
                    sys.stdout.write("Model Diverging: loglikelihood decreased\n")
                    break
                if curr_ll - prev_ll < self.tolerance:
                    sys.stdout.write("Model Diverged.\n")
                    break
            prev_ll = curr_ll

    def _next_iteration(self, corr_constant):
        ll, n_events, n_correct = 0.0, 0, 0
        for event in self.contexts:
            model_dist = self.prior.log_prior(event.context)
            model_dist = MaxentModel.evaluate(event.context, model_dist, self.eval_params)

            context = event.context
            for feature in context:
                if feature.index != -1 and\
                        self.pred_counts[feature.index] >= self.cutoff:
                    active_outcomes = self.model_expects[feature.index].outcomes
                    for aoi, oi in enumerate(active_outcomes):
                        self.model_expects[feature.index].update(
                                aoi, model_dist[oi] * feature.value * event.count)

            ll += log(model_dist[event.oid]) * event.count
            n_events += event.count
            max_dist = numpy.argmax(model_dist)
            if max_dist == event.oid:
                n_correct += event.count

        sys.stdout.write(".")

        for pi in range(self.n_preds):
            observed = self.observed_expects[pi].params
            model    = self.model_expects[pi].params
            outcomes = self.params[pi].outcomes
            for aoi, oi in enumerate(outcomes):
                if self.use_gaussian_smoothing:
                    pass
                else:
                    if model[aoi] == 0:
                        pass
                    self.params[pi].update(aoi, (log(observed[aoi]) - log(model[aoi]))/corr_constant)
                self.model_expects[pi].set(aoi, 0.0)

        self.eval_params.params = self.params
        sys.stdout.write(". loglikelihood=%f\t%f\n" % (ll, float(n_correct)/n_events))

        return ll

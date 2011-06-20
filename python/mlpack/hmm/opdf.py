import random
from mlpack.distribution import *
from mlpack.hmm.model import ObservationReal

class OpdfFactory(object):
    def __init__(self, config):
        self.config = config

    def factor(self):
        config = self.config

        opdf_type = config.get("opdf", "type")
        opdf = None
        if opdf_type == "integer":
            nb_entries = config.getint("opdf", "nb_entries")
            opdf = OpdfInteger([1.0 / nb_entries for i in range(nb_entries)])
        elif opdf_type == "gaussian":
            mean = config.getfloat("opdf", "mean")
            variance = config.getfloat("opdf", "variance")
            opdf = OpdfGaussian(mean, variance)
        return opdf

class OpdfInteger(object):
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def probability(self, o):
        return self.probabilities[o.value]

    def generate(self):
        rand = random.random()
        for i, p in enumerate(self.probabilities):
            rand -= p
            if rand < 0.0:
                return ObservationReal(i)
        return ObservationReal(len(self.probabilities)-1)

    def fit(self, co, weights=None):
        self.probabilities = [0.0 for p in self.probabilities]
        if weights is None:
            for o in co:
                self.probabilities[o.value] += 1.0
            csize = len(co)
            self.probabilities = map(lambda x: x / csize, self.probabilities)
        else:
            for i, o in enumerate(co):
                self.probabilities[o.value] += weights[i]

class OpdfGaussian(object):
    def __init__(self, mean=0.0, variance=1.0):
        self.distribution = GaussianDistribution(mean, variance)

    def probability(self, o):
        return self.distribution.probability(o.value)

    def generate(self):
        return ObservationReal(self.distribution.generate())

    def fit(self, co, weights=None):
        w = 1.0 / len(co)
        if weights is None:
            weights = [w for i in co]
        mean, var = 0.0, 0.0
        for i, o in enumerate(co):
            mean += o.value * weights[i]

        for i, o in enumerate(co):
            d = o.value - mean
            var += d * d * weights[i]

        self.distribution = GaussianDistribution(mean, var)

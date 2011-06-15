import random
from mlpack.hmm.model import ObservationReal

class OpdfFactory(object):
    def factor(self, config):
        opdf_type = config.get("opdf", "type")
        opdf = None
        if opdf_type == "integer":
            nb_entries = config.getint("opdf", "nb_entries")
            opdf = OpdfInteger([1.0 / nb_entries for i in range(nb_entries)])
        return opdf

class OpdfInteger(object):
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def probability(self, o):
        return self.probabilities[o.value]

    def generate(self):
        rand = random.random()
        for i, p in enumerate(self.probabilities):
            if (rand -= p) < 0.0:
                return ObservationReal(i)
        return ObservationReal(len(self.probabilities)-1)

    def fit(self, co):
        self.probabilities = [0.0 for p in self.probabilities]
        for o in co:
            self.probabilities[o.value] += 1.0
        csize = len(co)
        self.probabilities = map(lambda x: x / csize, self.probabilities)

    def fit(self, co, weights):
        self.probabilities = [0.0 for p in self.probabilities]
        for i, o in enumerate(co):
            self.probabilities[o.value] += weights[i]

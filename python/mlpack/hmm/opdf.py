import random
from mlpack.distribution import *
from mlpack.hmm.model import ObservationReal

class OpdfFactory(object):
    """
    Factory class for producing different output probability distribution
    functions. The type of distribution function is specified in the
    configuration file
    """
    def __init__(self, config):
        """
        """
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
    """
    Discrete integer output probability distribution.
    """
    def __init__(self, probabilities):
        """
        :type  probabilities: list
        :param probabilities: The table of probabilities to assign to discrete integer values
        """
        self.probabilities = probabilities

    def probability(self, o):
        """
        Return the probability corresponding to ``o.value``. This is simply return the assigned
        probability in the probability table.
        """
        return self.probabilities[o.value]

    def generate(self):
        """
        Randomly generate an observation based on the probabilities.
        """
        rand = random.random()
        for i, p in enumerate(self.probabilities):
            rand -= p
            if rand < 0.0:
                return ObservationReal(i)
        return ObservationReal(len(self.probabilities)-1)

    def fit(self, co, weights=None):
        """
        Computing the probability table based on the given observation list.
        If ``weights`` is ``None``, then the probabilities are uniformly
        distributed, otherwise, they are normalised to the sum of weights - i.e.
        :math:`p_{i} = weights_{i}/\sum_{j=1}^{N}{weights_{j}}`

        :type   co: list
        :param  co: The list of observations
        """
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
    """
    Gaussian based output probability distribution function
    """
    def __init__(self, mean=0.0, variance=1.0):
        """
        :type  mean: float
        :param mean: Mean of this gaussian distribution - :math:`\mu`
        :type  variance: float
        :param variance: Variance of this gaussian distribution - :math:`\sigma`
        """
        self.distribution = GaussianDistribution(mean, variance)

    def probability(self, o):
        """
        Probability of ``p(x = o.value)`` in this gaussian distribution - :math:`N(x=o.value|\mu=mean, \sigma=variance)`
        """
        return self.distribution.probability(o.value)

    def generate(self):
        """
        Generate a real number observation based on this gaussian distribution
        """
        return ObservationReal(self.distribution.generate())

    def fit(self, co, weights=None):
        """
        Compute the mean and variance of this gaussian distribution based on the observations
        and ``weights`` given. If ``weights`` is ``None``, then it is initialized to
        :math:`1.0/Len(Observations)`.

        The mean is computed as: :math:`\mu = \sum_{i=1}^{N}{observation * weights_{i}}`

        The variance is computed as: :math:`\sigma = \sum_{i=1}^{N}{observation * observation * weights_{i}}`

        :type   co: list
        :param  co: The list of observations
        """
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

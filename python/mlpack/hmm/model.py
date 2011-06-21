import copy
import math
import numpy

class CentroidReal(object):
    def __init__(self, o):
        """
        """
        self.value = o.value

    def reeval_add(self, e, v):
        self.value = ((self.value * len(v)) +\
                e.value) / (len(v) + 1.0)

    def reeval_remove(self, e, v):
        self.value = ((self.value * len(v)) -\
                e.value) / (len(v) - 1.0)

    def distance(self, e):
        return abs(e.value - self.value)

class CentroidVector(object):
    def __init__(self, o):
        """
        """
        self.value = copy.deepcopy(o)

    def reeval_add(self, e, v):
        values = e.value
        for i, v in enumerate(self.value):
            self.value[i] = ((self.value[i] * len(v)) + evalues[i]) / (len(v) + 1.0)

    def reeval_remove(self, e, v):
        values = e.value
        for i, v in enumerate(self.value):
            self.value[i] = ((self.value[i] * len(v)) - evalues[i]) / (len(v) - 1.0)

    def distance(self, e):
        diff = self.value - e
        return math.sqrt(sum(v * v for v in diff))

class ObservationReal(object):
    def __init__(self, value):
        """
        """
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    def factor(self):
        return CentroidReal(self)

class ObservationVector(object):
    def __init__(self, value):
        """
        """
        self.value = value

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        return iter(self.value)

    def __getitem__(self, i):
        return self.value[i]

    def __setitem__(self, i, v):
        self.value[i] = v

    def __add__(self, o):
        values = [sv + ov for sv, ov in zip(self, o)]
        s = ObservationVector(values)
        return s

    def __sub__(self, o):
        values = [sv - ov for sv, ov in zip(self, o)]
        s = ObservationVector(values)
        return s

    def __mul__(self, c):
        p = ObservationVector([v * c for v in self])
        return p

    def factor(self):
        return CentroidVector(self)

class HmmModel(object):
    def __init__(self, nb_states, opdf_factory=None):
        """
        """
        prob = 1.0 / nb_states
        self.pi = numpy.array([prob for i in range(nb_states)])
        self.a  = numpy.array([[prob for j in range(nb_states)] for i in range(nb_states)])
        if opdf_factory is not None:
            self.opdfs = [opdf_factory.factor() for i in range(nb_states)]
        else:
            self.opdfs = [None for i in range(nb_states)]

    def nb_states(self):
        return len(self.pi)

    def get_pi(self, i):
        return self.pi[i]

    def set_pi(self, i, p):
        self.pi[i] = p

    def get_opdf(self, i):
        return self.opdfs[i]

    def set_opdf(self, i, o):
        self.opdfs[i] = o

    def get_aij(self, i, j):
        return self.a[i][j]

    def set_aij(self, i, j, v):
        self.a[i][j] = v

    def copy(self):
        hmm = HmmModel(self.nb_states())
        hmm.pi = numpy.copy(self.pi)
        hmm.a  = numpy.copy(self.a)
        hmm.opdfs = copy.deepcopy(self.opdfs)
        return hmm

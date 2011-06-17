from math import log
import numpy

class UniformPrior(object):
	def log_prior(self, context):
		dist = numpy.array([self.r for i in range(self.n_outcomes)])
		return dist

	def set_labels(self, olabels, plables):
		self.n_outcomes = len(olabels)
		self.r = log(1.0 / self.n_outcomes)

from math import log

class UniformPrior(object):
	def log_prior(self, context):
		dist = [self.r for i in range(self.n_outcomes)]
		return dist

	def set_labels(self, olabels, plables):
		self.n_outcomes = len(olabels)
		self.r = log(1.0 / self.n_outcomes)

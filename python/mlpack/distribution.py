"""
Includes classes for the following list of distribution functions:
	1. Gaussian - M{N(mu, sigma) = (1/sqrt(2*PI)*sigma) * exp(-(x-mu)^2/2*sigma^2)}
	2. Exponential
	3. Poisson
"""

import math
import random

#: Log factorial table up to 29 for computing Poisson probabilities
log_factorial_table = (
		0.00000000000000000,   0.00000000000000000,   0.69314718055994531,
		1.79175946922805500,   3.17805383034794562,   4.78749174278204599,
		6.57925121201010100,   8.52516136106541430,  10.60460290274525023,
		12.80182748008146961,  15.10441257307551530,  17.50230784587388584,
		19.98721449566188615,  22.55216385312342289,  25.19122118273868150,
		27.89927138384089157,  30.67186010608067280,  33.50507345013688888,
		36.39544520803305358,  39.33988418719949404,  42.33561646075348503,
		45.38013889847690803,  48.47118135183522388,  51.60667556776437357,
		54.78472939811231919,  58.00360522298051994,  61.26170176100200198,
		64.55753862700633106,  67.88974313718153498,  71.25703896716800901
		)

C0 =  9.18938533204672742e-01
C1 =  8.33333333333333333e-02
C3 = -2.77777777777777778e-03
C5 =  7.93650793650793651e-04
C7 = -5.95238095238095238e-04

class Distribution(object):
	def dimension(self):
		"""
		For multivariate distributions, return the dimension of the
		variable. Otherwise, return 1
		"""
		return 1

	def generate(self):
		"""
		Generate a random number according to this distribution
		"""
		return 0.0

	def probability(self, n):
		"""
		Return the probability of n given this distribution
		"""
		return 0.0

class ExponentialDistribution(Distribution):
	"""
	Exponential probability distribution
	"""
	def __init__(self, rate):
		self.rate = rate

	def generate(self):
		return random.expovariate(self.rate)

	def probability(self, n):
		return self.rate * math.exp(-n * self.rate)

class GaussianDistribution(Distribution):
	"""
	Gaussian probability distribution
	"""
	def __init__(self, mean=0.0, variance=1.0):
		self.mean = mean
		self.variance = variance
		self.deviation = math.sqrt(variance)

	def generate(self):
		return random.gauss(self.mean, self.deviation)

	def probability(self, n):
		exp_arg = -0.5 * (n - self.mean) * (n - self.mean) / self.variance
		return pow(2.0 * math.pi * self.variance, -0.5) * math.exp(exp_arg)

class PoissonDistribution(Distribution):
	"""
	Poisson probability distribution
	"""
	def __init__(self, mean):
		self.mean = mean

	def generate(self):
		count, product = 0, 1.0
		elambda = math.exp(-self.mean)

		while product > elambda:
			product *= math.random()
			count += 1

		return count - 1

	def probability(self, n):
		return math.exp(n * math.log(self.mean) \
				- self._log_factorial(n) - self.mean)

	def _log_factorial(self, n):
		if n >= len(log_factorial_table):
			r = 1.0 / n
		return (n + 0.5) * math.log(n) - n + C0 \
				+ r * (C1 + r * r * (C3 + r * r * (C5 + r * r * C7)))

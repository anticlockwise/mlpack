LINEAR     = 0
POLYNOMIAL = 1
RBF        = 2
SIGMOID    = 3
CUSTOM     = 4

class KernelParameter:
	def __init__(self, kernel_type=LINEAR, poly_degree=3,
			rbf_gamma=1.0, coef_lin=1, coef_const=1,
			custom=""):
		self.kernel_type = kernel_type
		self.poly_degree = poly_degree
		self.rbf_gamma   = rbf_gamma
		self.coef_lin    = coef_lin
		self.coef_const  = coef_const
		self.custom      = custom

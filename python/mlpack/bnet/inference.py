import numpy
from mlpack.bnet.network import *

class Inference:
    def __init__(self, bnet, dpc):
        self.bnet = bnet
        self.dpc = dpc

    def inference(self, qvar_name=None):
        if self.dpc:
            index_qvar = self.bnet.index_of_variable(qvar_name)

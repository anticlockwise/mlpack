class Parameters(object):
    def __init__(self, params=[], outcomes=[]):
        self.params = params
        self.outcomes = outcomes

    def set(self, oi, param):
        self.params[oi] = param

    def update(self, oi, param):
        self.params[oi] += param

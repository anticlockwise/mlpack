class Model(object):
    def __init__(self, olabels, plabels):
        self.olabels = olabels
        self.plabels = plabels
        self.pmap = dict([(p, i) for i, p in enumerate(plabels)])

    def best_outcome(self, outcomes):
        best = 0
        for i, o in enumerate(outcomes):
            if outcomes[i] > outcomes[best]:
                best = i
        return self.olabels[best]

    def outcome(self, i):
        return self.olabels[i]

    def index(self, outcome):
        return self.olabels.index(outcome)

    def pred_index(self, pred):
        if pred in self.pmap:
            return self.pmap[pred]
        return -1

'''
File: events.py
Author: Rongzhou Shen <anticlockwise5@gmail.com>
Description: For representing an event instance in the training data
'''

import re

SPACES = re.compile(r"\s+")

def cmp_outcome(e1, e2):
    return e1.oid - e2.oid

class Feature(object):
    def __init__(self, name="", value=0.0):
        self.name = name
        self.value = value
        self.index = -1;

    def __add__(self, other):
        return Feature("", self.value + other.value)

class FeatureSet(object):
    def __init__(self):
        self.feat_map = {}
        self.attrs = {}

    def __len__(self):
        return len(self.feat_map)

    def __getitem__(self, key):
        return self.feat_map[key]

    def __setitem__(self, key, value):
        self.feat_map[key] = value

    def __contains__(self, item):
        return (item in self.feat_map)

    def __iter__(self):
        return self.feat_map.itervalues()

    def features(self):
        return self.feat_map.values()

    def attr(self, name, value=None):
        if value is not None:
            self.attrs[name] = value
        return self.attrs[name]

    def len_sq(self):
        keys = self.feat_map.keys()
        def sq(key):
            v = self.feat_map[key].value
            return v * v
        return reduce(lambda k, y: y + sq(k), keys)

class Event(object):
    def __init__(self, context=None, outcome=None, count=1):
        self.index = 0;
        self.context = context
        self.outcome = outcome
        self.count = count
        self.oid = -1

    def __cmp__(self, other):
        fmap1 = self.context
        fmap2 = other.context
        l_fmap1 = len(fmap1)
        l_fmap2 = len(fmap2)
        if l_fmap1 != l_fmap2:
            return l_fmap1 - l_fmap2

        for key in fmap1:
            if key not in fmap2:
                return 1
            f1 = fmap1[key]
            f2 = fmap2[key]
            if f1.value != f2.value:
                return f1.value - f2.value

        return cmp(self.outcome, other.outcome)

    def __eq__(self, other):
        return cmp(self, other) == 0

class SequenceEventStream(object):
    def __init__(self):
        self.events = []

    def __iter__(self):
        return iter(self.events)

    def add_event(self, event):
        self.events.append(event)

class BooleanEventStream(object):
    def __init__(self, filename, has_outcome=True):
        self.st_file = open(filename)

    def __iter__(self):
        for line in st_file:
            words = SPACES.split(line.strip())
            outcome = None
            if self.has_outcome:
                outcome = words[-1]
                words = words[:-1]
            fset = []
            for feat in words:
                name = feat
                value = 1.0
                fset.append(Feature(name, value))
            event = Event(fset, outcome, 1)
            yield event

class RealValueEventStream(object):
    def __init__(self, filename, has_outcome=True):
        self.st_file = open(filename)
        self.has_outcome = has_outcome

    def __iter__(self):
        for line in self.st_file:
            words = SPACES.split(line.strip())
            outcome = None
            if self.has_outcome:
                outcome = words[-1]
                words = words[:-1]
            fset = []
            for feat in words:
                name, value = feat.split("=")
                value = float(value)
                fset.append(Feature(name, value))
            event = Event(fset, outcome, 1)
            yield event

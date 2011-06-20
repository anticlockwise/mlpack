"""
Includes classes for reading and representing supervised-learning
training data.
"""

import re

#: Spaces for separating the data items
SPACES = re.compile(r"\s+")

def cmp_outcome(e1, e2):
    """
    Compares L{Event<mlpack.events.Event>} objects based on their
    outcome id numbers.

    @type  e1: L{Event<mlpack.events.Event>}
    @param e1: First event
    @type  e2: L{Event<mlpack.events.Event>}
    @param e2: Second event
    @rtype:    number
    @return:   -1 if e1 has a smaller outcome id, 1 if e1 has a bigger
               outcome id, otherwise 0
    """
    return e1.oid - e2.oid

class Feature(object):
    """
    Represents a predicate in the form: C{<B{pred_name}, B{value}, B{index}>}
    """
    def __init__(self, name="", value=0.0):
        self.name = name
        self.value = value
        self.index = -1;

    def __add__(self, other):
        return Feature("", self.value + other.value)

class FeatureSet(object):
    """
    Represents a list of L{Predicate<mlpack.events.Feature>}s and includes
    pythonic list data model methods for convenience. So you could do:

    >>> feat_set = FeatureSet()
    >>> feat_set["pred_name"] = 1.0
    >>> "pred_name" in feat_set
    True
    >>> len(feat_set)
    1
    >>> feat_set["pred_name"]
    1.0
    """
    def __init__(self):
        self.feat_map = {}
        self.attrs    = {}

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

    def attr(self, name, value=None):
        """
        Get/set an attribute on this predicate set.

        @type  name:  string
        @param name:  The name of the attribute
        @type  value: any
        @param value: The value of the attribute
        @rtype:       any
        @return:      The value of the attribute to be set/get
        """
        if value is not None:
            self.attrs[name] = value
        return self.attrs[name]

    def len_sq(self):
        """
        Compute the Euclidean length of this predicate vector -
        M{sum(p(i)*p(i)) i = 1..N}

        @rtype:  number
        @return: The Euclidean length of this vector.
        """
        keys = self.feat_map.keys()
        def sq(key):
            v = self.feat_map[key].value
            return v * v
        return reduce(lambda k, y: y + sq(k), keys)

class Event(object):
    """
    Represents a row of data in supervised-learning, normally
    in the form of C{<feature1> <feature2> <feature3> ... <outcome>}
    """
    def __init__(self, context=None, outcome=None, count=1):
        """
        Initialize an event.

        @type  context: L{FeatureSet<mlpack.events.FeatureSet>}
        @param context: The L{list of predicates<mlpack.events.FeatureSet>} given in the data
        @type  outcome: string
        @param outcome: The outcome label for this data item
        @type  count:   number
        @param count:   The number of times that this data item ocurred in the training data set
        """
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
    """
    Event stream class for manual creation of event data set
    """
    def __init__(self):
        self.events = []

    def __iter__(self):
        return iter(self.events)

    def add_event(self, event):
        """
        Add a data item to this stream.

        @type  event: L{Event<mlpack.events.Event>}
        @param event: The event to add
        """
        self.events.append(event)

class BooleanEventStream(object):
    """
    Class for reading event streams from a file in the format of:
    C{<feature1> <feature2> <feature3> ... <outcome>}
    """
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
    """
    Class for reading event streams from a file in the format of:
    C{<feature1=value1> <feature2=value2> <feature3=value3> ... <outcome>}
    """
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

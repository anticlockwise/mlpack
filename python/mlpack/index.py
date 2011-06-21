"""
Includes classes for indexing streamed events data for supervised-learning.
"""

from mlpack.events import *
import numpy

class DataIndexer(object):
    """
    Base class for all data indexers. Stores the list of events read,
    predicate labels encountered, outcome labels encountered, number of
    times each predicate occurred and the total number of events found.
    """
    def __init__(self):
        self.events = []
        self.plabels = []
        self.olabels = []
        self.pcounts = []
        self.n_events = 0

    def _update(self, context, pred_index, counter, cutoff):
        for feature in context:
            key = feature.name
            if key not in counter:
                counter[key] = 1
            else:
                counter[key] += 1

            if key not in pred_index:
                ind = len(pred_index)
                pred_index[key] = ind

    def _sort_n_merge(self, elist, sort=False):
        n_uniq_events = 1
        self.n_events = len(elist)
        if self.n_events <= 1:
            return n_uniq_events

        if sort:
            uniq_events = []
            elist.sort()
            print "Sorted."

            eprev = elist[0]
            uniq_events.append(eprev)
            for ecur in elist[1:]:
                if eprev == ecur:
                    uniq_events[n_uniq_events-1].count += ecur.count
                else:
                    ecur.index = n_uniq_events
                    uniq_events.append(ecur)
                    n_uniq_events += 1
                    eprev = ecur

            self.events = uniq_events
        else:
            ind = 0
            n_uniq_events = self.n_events
            self.events = elist
            for e in self.events:
                e.index = ind
                ind += 1

        return n_uniq_events

    def contexts(self):
        """
        Return the list of events indexed
        """
        return self.events

    def pred_labels(self):
        """
        Return the list of predicate labels in the data
        """
        return self.plabels

    def pred_counts(self):
        """
        Return the number of times each predicate is found
        """
        return self.pcounts

    def outcome_labels(self):
        """
        Return the list of outcome labels in the data
        """
        return self.olabels

    def num_events(self):
        """
        Return the number of events in the data
        """
        return self.n_events

class OnePassDataIndexer(DataIndexer):
    """
    Simple data indexer that does one pass over the data and assign each
    event, predicate and outcome with an integer index starting from 0.
    """
    def __init__(self, stream, cutoff=0, sort=True):
        """
        Initialize this indexer with an L{event stream<mlpack.events.EventStream>}.

        :type  stream: :py:class:`mlpack.event.EventStream`
        :param stream: The event stream to read unindexed events from
        :type  cutoff: number
        :param cutoff: The cutoff threshold for predicates - i.e. if the number of times that
                       a predicate occurred is smaller than this number, it will not be included.
        :type  sort:   boolean
        :param sort:   If True, the events will be sorted and merged (duplicated events will be
                       accumulated).
        """
        DataIndexer.__init__(self)
        print "Indexing events using cutoff %d" % cutoff
        print "Computing event counts and indexing..."
        self._compute_event_counts(stream, cutoff)
        print "DONE: %d in total" % len(self.events)
        print "Sorting and merging events..."
        self._sort_n_merge(self.events, sort)
        print "DONE: reduced to %d in total" % len(self.events)
        print "Indexing done."

    def _compute_event_counts(self, stream, cutoff):
        counter = {}
        oindex = {}
        pindex = {}

        for event in stream:
            self._update(event.context, pindex, counter, cutoff)

            if event.outcome not in oindex:
                ind = len(oindex)
                oindex[event.outcome] = ind

            event.oid = oindex[event.outcome]

            new_set = []
            old_set = event.context
            for feature in old_set:
                if feature.name in pindex:
                    feature.index = pindex[feature.name]
                    new_set.append(feature)

            event.context = new_set
            self.events.append(event)

        npreds = len(pindex)
        self.pcounts = numpy.zeros((npreds,))
        self.plabels = ["" for i in range(npreds)]
        self.olabels = ["" for i in range(len(oindex))]
        for pred, ind in pindex.items():
            self.pcounts[ind] = counter[pred]
            self.plabels[ind] = pred

        for o, ind in oindex.items():
            self.olabels[ind] = o

from mlpack.hmm.model import *
import csv

class ObservationsReader(object):
    """
    """
    def __init__(self):
        pass

    def read(self, filename):
        """
        Reads observation sequences from ``filename``. The file should
        be a CSV (comma-separate value) file with a sequence of observations
        per line. E.g.
        """
        reader = csv.reader(open(filename, 'rb'))
        sequences = []
        for row in reader:
            obs_row = [ObservationReal(int(d)) for d in row]
            sequences.append(obs_row)
        return sequences

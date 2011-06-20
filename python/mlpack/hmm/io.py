from mlpack.hmm.model import *
import csv

class ObservationsReader(object):
    def read(self, filename):
        reader = csv.reader(open(filename, 'rb'))
        sequences = []
        for row in reader:
            obs_row = [ObservationReal(int(d)) for d in row]
            sequences.append(obs_row)
        return sequences

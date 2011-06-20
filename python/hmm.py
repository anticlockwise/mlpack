from mlpack.hmm.model import *
from mlpack.hmm.learn import *
from mlpack.hmm.opdf import *
from mlpack.hmm.io import *
import sys
import ConfigParser as cg

if __name__ == '__main__':
    reader = ObservationsReader()
    sequences = reader.read(sys.argv[1])

    defaults = {
            "type": "integer",
            "nb_entries": "20"
            }
    config = cg.SafeConfigParser(defaults)
    config.add_section("opdf")

    factory = OpdfFactory(config)
    model = HmmModel(2, factory)
    learner = BaumWelchLearner()
    new_hmm = learner.learn(model, sequences)

    print new_hmm.pi
    print new_hmm.a

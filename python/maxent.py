from mlpack.maxent.trainer import *
from mlpack.index import *
from mlpack.events import *
from optparse import OptionParser
import pickle
import ConfigParser as cg
import sys

if __name__ == '__main__':
    usage = "usage: %prog [options] <input_file>"
    parser = OptionParser(usage=usage)
    parser.add_option("-c", "--config", dest="config",
            help="Configuration file", metavar="FILE")
    parser.add_option("-f", "--ftype", dest="ftype",
            help="Type of feature values: real,binary. Default: binary",
            default="binary", metavar="FEATURE_TYPE")
    parser.add_option("-t", "--train", action="store_true",
            dest="train", help="Training insteading of prediction")
    parser.add_option("-v", "--validation-file", dest="heldout",
            help="File containing validation/heldout events",
            metavar="FILE")
    parser.add_option("-m", "--model-file", dest="model",
            help="Output model file", metavar="FILE")

    options, args = parser.parse_args()

    defaults = {"iterations": "50",
            "cutoff": "0",
            "gis_prior": "uniform",
            "gis_simple_smoothing": "False",
            "gis_smoothing_observation": "0.1",
            "gis_gaussian_smoothing": "False",
            "gis_slack_param": "False",
            "gis_sigma": "2.0",
            "gis_tolerance": "0.0001"}
    config = cg.SafeConfigParser(defaults)
    config.add_section("maxent")

    if len(args) < 1:
        print "No input file given."
        parser.print_help()
        sys.exit(1)

    if options.model is None:
        print "No model file given."
        parser.print_help()
        sys.exit(1)

    stream = None
    if options.ftype == "real":
        stream = RealValueEventStream(args[0])
    else:
        stream = BooleanEventStream(args[0])

    if options.train:
        trainer = GISTrainer()
        if options.config:
            config.read(options.config)
        di = OnePassDataIndexer(stream, 0, True)
        model = trainer.train(di, config)
        pickle.dump(model, open(options.model, 'w'))
    else:
        model = pickle.load(open(options.model))
        for event in stream:
            fset = event.context
            for feature in fset:
                feature.index = model.pred_index(feature.name)
            probs = model.eval(fset)
            outcome = model.best_outcome(probs)
            if event.outcome is not None:
                print outcome, event.outcome
            else:
                print outcome

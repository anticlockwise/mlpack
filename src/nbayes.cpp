/*
 * =====================================================================================
 *
 *       Filename:  nbayes.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  02/06/11 14:51:01
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include <mlpack/nbayes.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

BOOST_CLASS_EXPORT(mlpack::NaiveBayesModel)

namespace mlpack {
    vector<double> NaiveBayesModel::eval(FeatureSet context) {
        shared_ptr<Distribution> dist = init_distribution(dist_type);
        int oi, n_outcomes = olabels.size();
        vector<double> probs(n_outcomes);

        for (oi = 0; oi < n_outcomes; oi++) {
            double prob = priors[oi];
            FeatureIterator fit;
            for (fit = context.begin(); fit != context.end(); fit++) {
                Feature &f = fit->second;
                if (pmap.find(f.name) != pmap.end()) {
                    int fid = pmap[f.name];
                    f.id = fid;
                    prob *= dist->prob(params, f, oi);
                }
            }
            probs[oi] = prob;
        }

        return probs;
    }

    NaiveBayesModel NaiveBayesTrainer::train(DataIndexer &di, ptree config) {
        string dist_type = config.get<string>("nbayes.dist_type", "uniform");
        EventSpace contexts = di.contexts();
        vector<string> olabels = di.outcome_labels();
        vector<string> plabels = di.pred_labels();

        dist = init_distribution(dist_type);

        int n_outcomes = olabels.size();
        int n_events = 0;

        cout << "Computing prior distribution P(C)...";
        vector<double> priors(n_outcomes);
        EventSpace::iterator eit;
        int oid;
        for (eit = contexts.begin(); eit != contexts.end(); eit++) {
            Event &ev = (*eit);
            oid = ev.oid;
            priors[oid] += ev.count;
            n_events += ev.count;
        }

        // Compute prior distribution p(c) parameters
        for (oid = 0; oid < n_outcomes; oid++) {
            priors[oid] /= n_events;
        }
        cout << "..done." << endl;

        cout << "Computing conditional probabilities P(X|C)...";
        // Compute p(x|c) parameters
        ParamMap pmap = dist->get_params(di, config);
        cout << "..done." << endl;

        NaiveBayesModel model(priors, olabels, plabels, pmap, dist_type);
        return model;
    }

    void NaiveBayesTrainer::set_heldout_data(EventSpace events) {

    }
}

using namespace std;
using namespace mlpack;
using boost::property_tree::ptree;
namespace po = boost::program_options;

int main(int argc, char** argv) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Print this help message")
        ("config,c", po::value<string>(), "Configuration file")
        ("feature_type,f", po::value<string>(), "Type of feature values: real,binary. Default:binary")
        ("train,t", "Training instead of prediction")
        ("validation-file,v", po::value<string>(), "File containing validation data/heldout events")
        ("input-file,i", po::value<string>(), "Input training/prediction data file")
        ("model-file,m", po::value<string>(), "Output model file");

    po::positional_options_description p;
    p.add("input-file", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
            .options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << endl;
        return 1;
    }

    string file;
    if (vm.count("input-file")) {
        file = vm["input-file"].as<string>();
    } else {
        cout << "No input file given." << endl;
        cout << "maxent [-options] <input_file>" << endl;
        cout << desc << endl;
        return 1;
    }

    string heldout_file;
    if (vm.count("validation-file")) {
        heldout_file = vm["validation-file"].as<string>();
    }

    string out_file;
    if (vm.count("model-file")) {
        out_file = vm["model-file"].as<string>();
    } else {
        cout << "No model file given." << endl;
        cout << "maxent [-options] <input_file>" << endl;
        cout << desc << endl;
        return 1;
    }

    if (vm.count("train")) {
        ptree pt;
        if (vm.count("config")) {
            read_ini(vm["config"].as<string>(), pt);
        }

        EventStream *stream;
        if (vm.count("feature_type")) {
            string type = vm["feature_type"].as<string>();
            if (type == "real") {
                stream = new RealValueFileEventStream(file);
            } else if (type == "binary") {
                stream = new FileEventStream(file);
            }
        } else {
            stream = new FileEventStream(file);
        }

        ofstream ofs(out_file.c_str());
        boost::archive::text_oarchive oa(ofs);

        OnePassDataIndexer di(*stream, 1, false);
        NaiveBayesTrainer trainer;
        NaiveBayesModel model = trainer.train(di, pt);
        oa << model;
    } else {
        NaiveBayesModel model;
        ifstream ifs(out_file.c_str());
        boost::archive::text_iarchive ia(ifs);
        ia >> model;

        EventStream *stream = NULL;
        if (vm.count("feature_type")) {
            string type = vm["feature_type"].as<string>();
            if (type == "real") {
                stream = new RealPredicateEventStream(file);
            } else if (type == "binary") {
                stream = new PredicateEventStream(file);
            }
        } else {
            stream = new PredicateEventStream(file);
        }

        while (stream->has_next()) {
            Event ev = stream->next();
            FeatureSet fset = ev.context;
            FeatureIterator fit;
            for (fit = fset.begin(); fit != fset.end(); fit++) {
                Feature &f = fit->second;
                f.id = model.pred_index(f.name);
            }
            vector<double> probs = model.eval(fset);
            vector<double>::iterator it;
            string o = model.best_outcome(probs);
            cout << o << endl;
        }
    }

    return 0;
}

/*
 * =====================================================================================
 *
 *       Filename:  maxent.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/30/2011 09:59:55
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

//#include "feature.hpp"
#include <mlpack/events.hpp>
#include <mlpack/index.hpp>
#include <mlpack/gistrainer.hpp>
#include <mlpack/model.hpp>
#include <mlpack/prior.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

using namespace std;
using namespace mlpack;
using boost::property_tree::ptree;
namespace po = boost::program_options;

int main(int argc, char** argv) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Print this help message")
        ("config,c", po::value<string>(), "Configuration file")
        ("feature_type,f", po::value<string>(), "Type of feature values: real,binary  default:binary")
        ("train,t", "Training instead of prediction")
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

        int cutoff = pt.get<int>("maxent.cutoff", 1);
        bool sort = pt.get<bool>("maxent.sort", true);
        OnePassDataIndexer di(*stream, cutoff, sort);
        GISTrainer trainer;
        GISModel model = trainer.train(di, NULL, pt);
        oa << model;
    } else {
        GISModel model;
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

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

BOOST_CLASS_EXPORT(mlpack::NaiveBayesModel)

namespace mlpack {
    void NaiveBayesCommand::init_trainer() {
        trainer = shared_ptr<NaiveBayesTrainer>(new NaiveBayesTrainer);
    }

    void NaiveBayesCommand::save_model(string fname, DataIndexer &di, ptree config) {
        NaiveBayesModel model = trainer->train(di, config);
        ofstream ofs(fname.c_str());
        boost::archive::text_oarchive oa(ofs);
        oa << model;
    }

    Model &NaiveBayesCommand::load_model(string fname) {
        ifstream ifs(fname.c_str());
        boost::archive::text_iarchive ia(ifs);
        ia >> model;
        return model;
    }

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

using namespace mlpack;

int main(int argc, char** argv) {
    NaiveBayesCommand cmd;
    return cmd.execute(argc, argv);
}

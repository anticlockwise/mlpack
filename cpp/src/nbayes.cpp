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
    vector<double> NaiveBayesModel::eval(FeatureSet context) {

    }

    NaiveBayesModel NaiveBayesTrainer::train(DataIndexer &di, ptree config) {
        string dist_type = config.get<string>("nbayes.dist_type", "uniform");
        EventSpace contexts = di.contexts();
        vector<string> olabels = di.outcome_labels();
        vector<string> plabels = di.pred_labels();

        dist = init_distribution(dist_type);

        int n_outcomes = olabels.size();
        int n_events = 0;

        vector<double> priors(n_outcomes);
        EventSpace::iterator eit;
        int oid;
        for (eit = contexts.begin(); eit != contexts.end(); eit++) {
            Event &ev = (*eit);
            oid = ev.oid;
            priors[oid] += ev.count;
            n_events += ev.count;
        }

        for (oid = 0; oid < n_outcomes; oid++) {
            priors[oid] /= n_events;
        }

        ParamMap pmap = dist->get_params(di, config);

        NaiveBayesModel model(priors, olabels, plabels, pmap, dist_type);
        return model;
    }

    void NaiveBayesTrainer::set_heldout_data(EventSpace events) {

    }
}

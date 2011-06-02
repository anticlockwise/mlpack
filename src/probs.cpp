/*
 * =====================================================================================
 *
 *       Filename:  probs.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  02/06/11 15:29:30
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include <mlpack/probs.hpp>

namespace mlpack {
    double GaussianDistribution::prob(ParamMap pmap, Feature &feat, int oid) {
        vector<double> params = pmap[make_pair(feat.id, oid)];
        double mean = params[0];
        double var = params[1];
        const double PI = 3.14159265358979323846;

        // Compute corresponding gaussian density
        // f(x) = [1/sqrt(2*PI*var)]*e^[-(x-mu)*(x-mu)/(2*var)]
        double sigma = 2 * var;
        double x_mu = feat.value - mean; // x - mu
        double x_mu_sq = x_mu * x_mu;
        double e = -x_mu_sq / sigma;

        double fx = (1.0 / sqrt(sigma*PI)) * exp(e);
        return fx;
    }

    ParamMap GaussianDistribution::get_params(DataIndexer &di, ptree config) {
        EventSpace events = di.contexts();
        vector<string> plabels = di.pred_labels();
        vector<string> olabels = di.outcome_labels();
        int n_preds = plabels.size();
        int n_outcomes = olabels.size();

        vector<vector<int> > pred_counts(n_preds, vector<int>(n_outcomes));
        vector<vector<double> > means(n_preds, vector<double>(n_outcomes));
        vector<vector<double> > vars(n_preds, vector<double>(n_outcomes));

        EventSpace::iterator eit;
        FeatureSet fset;
        FeatureIterator fit;
        for (eit = events.begin(); eit != events.end(); eit++) {
            Event &ev = (*eit);
            fset = ev.context;
            for (fit = fset.begin(); fit != fset.end(); fit++) {
                Feature &f = (*fit).second;
                pred_counts[f.id][ev.oid] += ev.count;
                means[f.id][ev.oid] += f.value * ev.count;
                vars[f.id][ev.oid] += f.value * f.value * ev.count;
            }
        }

        ParamMap pmap;
        int pi, oi;
        for (oi = 0; oi < n_outcomes; oi++) {
            for (pi = 0; pi < n_preds; pi++) {
                double mean = means[pi][oi] / pred_counts[pi][oi];
                double var = vars[pi][oi] / pred_counts[pi][oi] - mean * mean;
                vector<double> params(2);
                params[0] = mean;
                params[1] = var;
                pmap[make_pair(pi, oi)] = params;
            }
        }

        return pmap;
    }

    double UniformDistribution::prob(ParamMap pmap, Feature &feat, int oid) {
        vector<double> probs;
        Predicate p = make_pair(feat.id, oid);
        probs = pmap[p];
        return probs[0];
    }

    ParamMap UniformDistribution::get_params(DataIndexer &di, ptree config) {
        EventSpace events = di.contexts();
        vector<string> plabels = di.pred_labels();
        vector<string> olabels = di.outcome_labels();
        int n_preds = plabels.size();
        int n_outcomes = olabels.size();
        EventSpace::iterator eit;
        FeatureSet fset;
        FeatureIterator fit;
        vector<vector<int> > pred_counts(n_preds, vector<int>(n_outcomes));
        vector<int> pcount(n_outcomes);

        for (eit = events.begin(); eit != events.end(); eit++) {
            Event &ev = (*eit);
            fset = ev.context;
            pcount[ev.oid] += fset.size() * ev.count;
            for (fit = fset.begin(); fit != fset.end(); fit++) {
                Feature &f = (*fit).second;
                pred_counts[f.id][ev.oid] += ev.count;
            }
        }

        ParamMap pmap;
        int pi, oi;
        // Compute p(f|c) - Probably of a feature occuring given an outcome
        // Apply Laplace smoothing/add one smoothing
        for (oi = 0; oi < n_outcomes; oi++) {
            int pc = pcount[oi] + n_preds;
            for (pi = 0; pi < n_preds; pi++) {
                int poc = pred_counts[pi][oi] + 1;
                vector<double> params;
                params.push_back((double)poc/pc);
                pmap[make_pair(pi, oi)] = params;
            }
        }

        return pmap;
    }
}

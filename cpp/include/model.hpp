/*
 * =====================================================================================
 *
 *       Filename:  model.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  31/05/11 14:21:53
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <algorithm>
#include "feature.hpp"
#include "prior.hpp"
#include "params.hpp"

using namespace std;

class Model {
    public:
        virtual vector<double> eval(FeatureSet context) = 0;

        virtual string best_outcome(vector<double> outcomes) = 0;

        virtual string outcome(int i) = 0;

        virtual int index(string out) = 0;
};

class GISModel : public Model {
    protected:
        vector<string> olabels;
        MaxentParameters *maxent_params;
        Prior *prior;

    public:
        GISModel(vector<Parameters> params, vector<string> pls, vector<string> ols) {
            prior = new UniformPrior();
            maxent_params = new MaxentParameters(params, 0.0, 1.0, ols.size());
            olabels = ols;
        }

        ~GISModel() {
            delete prior;
            delete maxent_params;
        }

        static vector<double> eval(FeatureSet context, vector<double> &prior,
                MaxentParameters model) {
            vector<Parameters> params = model.params;
            vector<int> n_feats(model.n_outcomes);
            vector<int> active_outcomes;
            vector<double> active_params;
            double value = 1.0;
            int ai, n_outcomes, oid;
            FeatureIterator fit;
            for (fit = context.begin(); fit != context.end(); fit++) {
                Feature f = (*fit).second;
                if (f.id != -1) {
                    Parameters &p = params[f.id];
                    active_outcomes = p.outcomes;
                    active_params = p.params;
                    value = f.value;
                    n_outcomes = active_outcomes.size();
                    for (ai = 0; ai < n_outcomes; ai++) {
                        oid = active_outcomes[ai];
                        n_feats[oid]++;
                        prior[oid] += active_params[ai] * value;
                    }
                }
            }

            double normal = 0.0;
            for (oid = 0; oid < model.n_outcomes; oid++) {
                if (model.corr_param != 0) {
                    prior[oid] = exp(prior[oid] * model.const_inverse
                            + ((1.0 - ((double)n_feats[oid] / model.corr_constant))
                                * model.corr_param));
                } else {
                    prior[oid] = exp(prior[oid] * model.const_inverse);
                }
                normal += prior[oid];
            }

            for (oid = 0; oid < model.n_outcomes; oid++) {
                prior[oid] /= normal;
            }

            return prior;
        }

        vector<double> eval(FeatureSet context);
        string best_outcome(vector<double> outcomes);
        string outcome(int i);
        int index (string out);
};

#endif

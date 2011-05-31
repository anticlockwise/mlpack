/*
 * =====================================================================================
 *
 *       Filename:  prior.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/30/2011 12:49:32
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include "prior.hpp"

void UniformPrior::log_prior(vector<double> &dist, FeatureSet &context) {
    dist.resize(n_outcomes);
    int i;
    for (i = 0; i < n_outcomes; i++) {
        dist[i] = r;
    }
}

void UniformPrior::set_labels(vector<string> outcome_labels,
        vector<string> pred_labels) {
    n_outcomes = outcome_labels.size();
    r = log(1.0 / n_outcomes);
}

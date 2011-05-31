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

UniformPrior::UniformPrior(vector<string> o) {
    outcomes = o;
    int n_outcomes = outcomes.size();
    r = log(1.0 / n_outcomes);
}

void UniformPrior::log_prior(map<string, double> dist, FeatureSet &context) {
    vector<string>::iterator it;
    for (it = outcomes.begin(); it != outcomes.end(); it++) {
        dist[*it] = r;
    }
}

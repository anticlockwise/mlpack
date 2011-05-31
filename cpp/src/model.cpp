/*
 * =====================================================================================
 *
 *       Filename:  model.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  31/05/11 14:38:56
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */
#include "model.hpp"

vector<double> GISModel::eval(FeatureSet context) {
    vector<double> dist(olabels.size());
    prior->log_prior(dist, context);
    return GISModel::eval(context, dist, *maxent_params);
}

string GISModel::best_outcome(vector<double> outcomes) {
    int best = 0;
    int i, n = outcomes.size();
    for (i = 1; i < n; i++) {
        if (outcomes[i] > outcomes[best])
            best = i;
    }
    return olabels[best];
}

string GISModel::outcome(int i) {
    return olabels[i];
}

int GISModel::index(string out) {
    vector<string>::iterator it = find(olabels.begin(), olabels.end(), out);
    if (it != olabels.end()) {
        return it - olabels.begin();
    } else {
        return -1;
    }
}

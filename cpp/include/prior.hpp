/*
 * =====================================================================================
 *
 *       Filename:  prior.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/30/2011 12:41:58
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#ifndef PRIOR_H
#define PRIOR_H

#include <map>
#include <vector>
#include <iterator>
#include <cmath>
#include "events.hpp"

using namespace std;

class Prior {
    public:
        virtual void log_prior(vector<double> &dist, FeatureSet &context) = 0;

        virtual void set_labels(vector<string> outcome_labels, vector<string> pred_labels) = 0;
};

class UniformPrior : public Prior {
    int n_outcomes;
    double r;

    public:
        void log_prior(vector<double> &dist, FeatureSet &context);

        void set_labels(vector<string> outcome_labels, vector<string> pred_labels);
};

#endif

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
        virtual void log_prior(map<string, double> dist, FeatureSet &context) = 0;
};

class UniformPrior : public Prior {
    vector<string> outcomes;
    double r;

    public:
        UniformPrior(vector<string> o);

        void log_prior(map<string, double> dist, FeatureSet &context);
};

#endif

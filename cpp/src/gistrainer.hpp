/*
 * =====================================================================================
 *
 *       Filename:  gistrainer.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/30/2011 20:05:45
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */
#ifndef GISTRAINER_H
#define GISTRAINER_H

#include "trainer.hpp"
#include "index.hpp"

const double NEAR_ZERO = 0.01;
const double LL_THRESHOLD = 0.0001;

class GISModel {
};

class GISTrainer : public Trainer<GISModel> {
    bool use_simple_smoothing;

    bool use_slack_param;

    bool use_gaussian_smoothing;

    double sigma;

    double smoothing_observation;

    double cf_observed_expect;

    int n_uniq_events;

    int n_preds;

    int n_outcomes;

    int cutoff;

    EventSpace contexts;

    vector<string> outcome_labels;

    vector<string> pred_labels;

    Prior *prior;

    public:
    GISTrainer() {
        use_simple_smoothing = false;
        use_slack_param = false;
        use_gaussian_smoothing = false;
        sigma = 2.0;
        smoothing_observation = 0.1;
    }

    GISModel train(EventSpace events, ptree pt);
};

#endif

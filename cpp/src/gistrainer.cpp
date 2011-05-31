/*
 * =====================================================================================
 *
 *       Filename:  gistrainer.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/30/2011 20:08:43
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */
#include "gistrainer.hpp"

GISModel GISTrainer::train(DataIndexer &di, Prior *p, ptree pt) {
    int iterations = pt.get<int>("iterations");
    cutoff = pt.get<int>("cutoff");
    contexts = di.contexts();
    pred_counts = di.pred_counts();
    n_uniq_events = contexts.size();
    prior = p;

    int corr_constant = 1;
    EventSpace::iterator it;
    for (it = contexts.begin(); it != contexts.end(); it++) {
        double cl = 0.0;
        FeatureIterator fit;
        FeatureSet fset = (*it).context;
        for (fit = fset.begin(); fit != fset.end(); fit++) {
            cl += (*fit).second.value;
        }

        if (cl > corr_constant) {
            corr_constant = (int)ceil(cl);
        }
    }

    outcome_labels = di.outcome_labels();
    pred_labels = di.pred_labels();
    n_outcomes = outcome_labels.size();
    n_preds = pred_labels.size();
    prior->set_labels(outcome_labels, pred_labels);

    Matrix pred_count(n_preds, vector<double>(n_outcomes));
    int ei = 0, pi = 0, oi = 0;
    for (; pi < n_preds; pi++) {
        for (; oi < n_outcomes; oi++) {
            pred_count[pi][oi] = 0;
        }
    }

    ei = pi = oi = 0;
    FeatureIterator fit;
    FeatureSet fset;
    Feature f;
    Event ev;
    for (it = contexts.begin(); it != contexts.end(); it++, ei++) {
        ev = (*it);
        fset = ev.context;
        for (fit = fset.begin(); fit != fset.end(); fit++, pi++) {
            f = fit->second;
            pred_count[f.id][ev.oid] += ev.count * f.value;
        }
    }

    params.resize(n_preds);
    model_expects.resize(n_preds);
    observed_expects.resize(n_preds);

    vector<int> active_outcomes(n_outcomes);
    vector<int> outcome_pattern;
    vector<int> all_outcomes(n_outcomes);

    for (oi = 0; oi < n_outcomes; oi++) {
        all_outcomes[oi] = oi;
    }
    int n_active_outcomes = 0;
    for (pi = 0; pi < n_preds; pi++) {
        n_active_outcomes = 0;
        int aoi;
        if (use_simple_smoothing) {
            n_active_outcomes = n_outcomes;
            outcome_pattern = all_outcomes;
        } else {
            for (oi = 0; oi < n_outcomes; oi++) {
                if (pred_count[pi][oi] > 0 && pred_counts[pi] >= cutoff) {
                    active_outcomes[n_active_outcomes] = oi;
                    n_active_outcomes++;
                }
            }
            outcome_pattern.resize(n_active_outcomes);
            for (aoi = 0; aoi < n_active_outcomes; aoi++) {
                outcome_pattern[aoi] = active_outcomes[aoi];
            }
        }

        Parameters &p = params[pi];
        update(p, outcome_pattern, n_active_outcomes);
        p = model_expects[pi];
        update(p, outcome_pattern, n_active_outcomes);
        p = observed_expects[pi];
        update(p, outcome_pattern, n_active_outcomes);

        for (aoi = 0; aoi < n_active_outcomes; aoi++) {
            oi = outcome_pattern[aoi];
            p = params[pi];
            p.params[aoi] = 0.0;
            p = model_expects[pi];
            p.params[aoi] = 0.0;
            p = observed_expects[pi];
            if (pred_count[pi][oi] > 0) {
                p.params[aoi] = pred_count[pi][oi];
            } else if (use_simple_smoothing) {
                p.params[aoi] = smoothing_observation;
            }
        }
    }

    if (use_slack_param) {
    }

    num_feats.resize(n_outcomes);

    eval_params = new MaxentParameters(params, 0.0, 1.0, n_outcomes);
    find_params(iterations, corr_constant);
}

void GISTrainer::find_params(int iterations, int corr_constant) {
    double prev_ll = 0.0;
    double curr_ll = 0.0;
    int i;

    for (i = 0; i < iterations; i++) {
        curr_ll = next_iteration(corr_constant);
        if (i > 0) {
            if (prev_ll > curr_ll) {
                break;
            }
            if (curr_ll - prev_ll < LL_THRESHOLD) {
                break;
            }
        }
        prev_ll = curr_ll;
    }
}

double GISTrainer::next_iteration(int corr_constant) {
    vector<double> model_dist(n_outcomes);
    double ll = 0.0;
    int n_events = 0;
    int n_correct = 0;
    EventSpace::iterator eit;
    for (eit = contexts.begin(); eit != contexts.end(); eit++) {
        Event ev = (*eit);
        prior->log_prior(model_dist, ev.context);
        GISModel::eval(ev.context, model_dist, *eval_params);

        FeatureSet fset = ev.context;
        FeatureIterator fit;
        for (fit = fset.begin(); fit != fset.end(); fit++) {
            Feature f = (*fit).second;
            if (f.id != -1 && pred_counts[f.id] >= cutoff) {
                vector<int> active_outcomes = model_expects[f.id].outcomes;
                int aoi, n_ao = active_outcomes.size();
                for (aoi = 0; aoi < n_ao; aoi++) {
                    int oi = active_outcomes[aoi];
                    Parameters &p = model_expects[f.id];
                    p.update(aoi, model_dist[oi] * f.value * ev.count);
                }
            }
        }

        ll += log(model_dist[ev.oid]) * ev.count;
        n_events += ev.count;
    }

    int pi, aoi, n_out;
    vector<double> observed;
    vector<double> model;
    vector<int> outcomes;
    for (pi = 0; pi < n_preds; pi++) {
        observed = observed_expects[pi].params;
        model = model_expects[pi].params;
        outcomes = params[pi].outcomes;
        for (aoi = 0; aoi < n_out; aoi++) {
            if (use_gaussian_smoothing) {

            } else {
                if (model[aoi] == 0) {
                }
                params[pi].update(aoi, (log(observed[aoi]) - log(model[aoi]))/corr_constant);
            }
            model_expects[pi].set(aoi, 0.0);
        }
    }

    return ll;
}

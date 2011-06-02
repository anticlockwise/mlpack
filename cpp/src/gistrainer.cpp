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
#include <mlpack/gistrainer.hpp>

namespace mlpack {
    void GISTrainer::set_heldout_data(EventSpace events) {

    }

    MaxentModel GISTrainer::train(DataIndexer &di, ptree pt) {
        // Initialize parameters from configuration file
        int iterations = pt.get<int>("maxent.iterations", 15);
        cutoff = pt.get<int>("maxent.cutoff", 1);
        string prior_type = pt.get<string>("maxent.gis.prior", "uniform");
        use_simple_smoothing = pt.get<bool>("maxent.gis.simple_smoothing", false);
        smoothing_observation = pt.get<double>("maxent.gis.smoothing_observation", 0.1);
        use_gaussian_smoothing = pt.get<bool>("maxent.gis.guassian_smoothing", false);
        use_slack_param = pt.get<bool>("maxent.gis.slack_param", false);
        sigma = pt.get<double>("maxent.gis.sigma", 2.0);
        tolerance = pt.get<double>("maxent.gis.tolerance", 0.0001);

        if (prior_type == "uniform") {
            prior = new UniformPrior();
        }

        cout << "Incorporating indexed data for training..." << endl;

        contexts = di.contexts();
        pred_counts = di.pred_counts();
        n_uniq_events = contexts.size();

        // Determine correction constant
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

        cout << "done." << endl;

        outcome_labels = di.outcome_labels();
        pred_labels = di.pred_labels();
        n_outcomes = outcome_labels.size();
        n_preds = pred_labels.size();
        // Initialize prior distribution
        prior->set_labels(outcome_labels, pred_labels);

        cout << "\tNumber of predicates: " << n_preds << endl;
        cout << "\t    Number of events: " << n_uniq_events << endl;
        cout << "\t  Number of outcomes: " << n_outcomes << endl;

        // Initialize predicate/outcome frequence table
        // pred_count: predicate/outcome frequency table - how many times a predicate
        // has been seen with a particular outcome
        Matrix pred_count(n_preds, vector<double>(n_outcomes));
        int ei = 0, pi = 0, oi = 0;
        for (; pi < n_preds; pi++) {
            for (; oi < n_outcomes; oi++) {
                pred_count[pi][oi] = 0;
            }
        }

        // Populate predicate/outcome frequency table
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

            // Initilize parameters for predicate pi
            update(&params[pi], outcome_pattern, n_active_outcomes);
            update(&model_expects[pi], outcome_pattern, n_active_outcomes);
            update(&observed_expects[pi], outcome_pattern, n_active_outcomes);

            Parameters *pa;
            for (aoi = 0; aoi < n_active_outcomes; aoi++) {
                oi = outcome_pattern[aoi];
                pa = &params[pi];
                pa->set(aoi, 0.0);
                pa = &model_expects[pi];
                pa->set(aoi, 0.0);
                pa = &observed_expects[pi];
                if (pred_count[pi][oi] > 0) {
                    pa->set(aoi, pred_count[pi][oi]);
                } else if (use_simple_smoothing) {
                    pa->set(aoi, smoothing_observation);
                }
            }
        }

        if (use_slack_param) {
        }

        num_feats.resize(n_outcomes);

        cout << "...done." << endl;
        eval_params = new MaxentParameters(params, 0.0, 1.0, n_outcomes);
        cout << "Computing model parameters..." << endl;
        find_params(iterations, corr_constant);

        MaxentModel model(params, pred_labels, outcome_labels, prior);

        return model;
    }

    void GISTrainer::find_params(int iterations, int corr_constant) {
        double prev_ll = 0.0;
        double curr_ll = 0.0;
        int i;

        cout << "Performing " << iterations << " iterations" << endl;
        for (i = 0; i < iterations; i++) {
            cout << (i + 1) << ":  ";
            curr_ll = next_iteration(corr_constant);
            if (i > 0) {
                if (prev_ll > curr_ll) {
                    cout << "Model Diverging: loglikelihood decreased" << endl;
                    break;
                }
                if (curr_ll - prev_ll < tolerance) {
                    cout << "Model Diverged." << endl;
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
            model_dist = MaxentModel::eval(ev.context, model_dist, *eval_params);

            FeatureSet fset = ev.context;
            FeatureIterator fit;
            for (fit = fset.begin(); fit != fset.end(); fit++) {
                Feature f = (*fit).second;
                if (f.id != -1 && pred_counts[f.id] >= cutoff) {
                    vector<int> active_outcomes = model_expects[f.id].outcomes;
                    int aoi, n_ao = active_outcomes.size();
                    for (aoi = 0; aoi < n_ao; aoi++) {
                        int oi = active_outcomes[aoi];
                        Parameters *p = &model_expects[f.id];
                        p->update(aoi, model_dist[oi] * f.value * ev.count);
                    }
                }
            }

            ll += log(model_dist[ev.oid]) * ev.count;
            n_events += ev.count;
            int max = 0, oi;
            for (oi = 1; oi < n_outcomes; oi++) {
                if (model_dist[oi] > model_dist[max])
                    max = oi;
            }
            if (max == ev.oid) {
                n_correct += ev.count;
            }
        }

        cout << ".";

        int pi, aoi, n_out;
        vector<double> observed;
        vector<double> model;
        vector<int> outcomes;
        for (pi = 0; pi < n_preds; pi++) {
            observed = observed_expects[pi].params;
            model = model_expects[pi].params;
            outcomes = params[pi].outcomes;
            n_out = outcomes.size();
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
        eval_params->params = params;
        cout << ". loglikelihood=" << ll << "\t" << ((double)n_correct/n_events) << endl;

        return ll;
    }
}

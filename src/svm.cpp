/*
 * =====================================================================================
 *
 *       Filename:  svm.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  03/06/11 16:35:09
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include <mlpack/svm.hpp>

namespace mlpack {
    SVCQMatrix::SVCQMatrix(EventSpace events, shared_ptr<SVMParameters> params): QMatrix(events, params) {
        contexts = events;
        int n_events = events.size();
        int i;
        qd.reserve(n_events);
        for (i = 0; i < n_events; i++) {
            FeatureSet &fset = events[i].context;
            qd[i] = kernel->eval(fset, fset);
        }
    }

    vector<double> SVCQMatrix::get_q(int i , int l) {
        
    }

    vector<double> SVCQMatrix::get_qd() {
        return qd;
    }

    void Solver::solve(int ne, shared_ptr<QMatrix> q,
            vector<double> p, EventSpace &events,
            vector<double> alpha, double c_p, double c_n,
            double eps, int shrinking) {
        cp = c_p;
        cn = c_n;
        alpha_status.reserve(ne);
        active_set.reserve(ne);
        gradient.reserve(ne);
        g_bar.reserve(ne);

        int i, j;
        for (i = 0; i < ne; i++) {
            update_alpha_status(alpha[i], i, events[i].oid);
            active_set[i] = i;
            gradient[i] = p[i];
            g_bar[i] = 0;
        }

        for (i = 0; i < ne; i++) {
            if (!is_lower_bound(i)) {

            }
        }
    }

    vector<double> SVMModel::eval(FeatureSet context) {
        
    }

    void SVMTrainer::group_classes(EventSpace &events, vector<int> &start,
            vector<int> &count, int n_classes) {
        EventSpace::iterator eit = events.begin();
        while (eit != events.end()) {
            Event &e = (*eit);
            count[e.oid]++;
            eit++;
        }

        int i = 0;
        start[i] = 0;
        for (i = 1; i < n_classes; i++) {
            start[i] = start[i-1] + count[i-1];
        }
    }

    DecisionFunction SVMTrainer::train_single_class(EventSpace &subspace,
            shared_ptr<SVMParameters> params, int wp, int wn) {
        vector<double> alpha(subspace.size());
        switch (params->svm_type) {
            case C_SVC:
                break;
            case NU_SVC:
                break;
            case ONE_CLASS:
                break;
            case EPSILON_SVR:
                break;
            case NU_SVR:
                break;
        }
    }

    SVMModel SVMTrainer::train(DataIndexer &di, ptree config) {
        SVMModel model;

        learn_params = shared_ptr<SVMParameters>(new SVMParameters);
        init_params(learn_params, config);

        if (learn_params->svm_type == ONE_CLASS
                || learn_params->svm_type == EPSILON_SVR
                || learn_params->svm_type == NU_SVR) {
            // TODO: To be implemented
        } else { // Classification
            vector<string> olabels = di.outcome_labels();
            int n_classes = olabels.size();
            vector<double> weighted_c(n_classes);
            vector<bool> nonzero(di.num_events(), false);
            EventSpace events = di.contexts();

            vector<int> start(n_classes), count(n_classes);
            sort(events.begin(), events.end(), cmp_outcome);
            group_classes(events, start, count, n_classes);

            int i, j, k;
            for (i = 0; i < n_classes; i++) {
                weighted_c[i] = learn_params->c;
            }

            vector<DecisionFunction> f(n_classes*(n_classes-1)/2);
            vector<double> probA, probB;
            if (learn_params->probability == 1) {
                probA.reserve(n_classes*(n_classes-1)/2);
                probB.reserve(n_classes*(n_classes-1)/2);
            }

            int p = 0;
            for (i = 0; i < n_classes; i++) {
                for (j = i + 1; j < n_classes; j++) {
                    EventSpace sub_space;
                    int si = start[i], sj = start[j];
                    int ci = count[i], cj = count[j];
                    sub_space.reserve(ci + cj);
                    for (k = 0; k < ci; k++) {
                        sub_space[k] = events[si+k];
                        sub_space[k].oid = 1;
                    }
                    for (k = 0; k < cj; k++) {
                        sub_space[ci+k] = events[sj+k];
                        sub_space[ci+k].oid = -1;
                    }

                    if (learn_params->probability == 1) {
                        // TODO implement probability calculation
                    }

                    f[p] = train_single_class(sub_space, learn_params, weighted_c[i], weighted_c[j]);
                }
            }
        }
    }

    void SVMTrainer::set_heldout_data(EventSpace events) {
        
    }
}

int main(int argc, char** argv) {
    return 0;
}

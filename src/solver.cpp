/*
 * =====================================================================================
 *
 *       Filename:  solver.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  13/06/11 11:58:24
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include <mlpack/solver.hpp>

namespace mlpack {
    void Solver::calculate_rho(AlphaModel &model) {
        double r = 0.0;
        int nr_free = 0;
        double ub = numeric_limits<double>::max();
        double lb = numeric_limits<double>::min();
        double sum_free = 0.0;

        vector<SolutionVector>::iterator it;
        for (it = active.begin(); it != active.end(); it++) {
            SolutionVector &sv = *it;
            Event &e = sv.event;
            double yg = (e.oid == 1 ? 1.0 : -1.0) * sv.g;

            if (sv.is_lower_bound()) {
                if (e.oid == 1) {
                    ub = min(ub, yg);
                } else {
                    lb = max(lb, yg);
                }
            } else if (sv.is_upper_bound()) {
                if (e.oid == 0) {
                    ub = min(ub, yg);
                } else {
                    lb = max(lb, yg);
                }
            } else {
                ++nr_free;
                sum_free += yg;
            }
        }

        if (nr_free > 0) {
            r = sum_free / nr_free;
        } else {
            r = (ub + lb) / 2;
        }

        model.rho = r;
    }

    void Solver::init_active_set() {
        active = all_examples;
        q_sva.resize(n_examples, 0.0);
        q_svb.resize(n_examples, 0.0);
    }

    int Solver::optimize() {
        q->init_ranks(all_examples);
        vector<SolutionVector>::iterator it, b_it;
        for (it = all_examples.begin(); it != all_examples.end(); it++) {
            it->update_alpha_status(cp, cn);
        }

        init_active_set();

        for (it = all_examples.begin(); it != all_examples.end(); it++) {
            it->g = it->linear_term;
            it->g_bar = 0.0;
        }

        for (it = all_examples.begin(); it != all_examples.end(); it++) {
            if (!it->is_lower_bound()) {
                q->get_q(*it, active, q_sva);
                for (b_it = all_examples.begin(); b_it != all_examples.end(); b_it++) {
                    b_it->g += it->alpha * q_sva[b_it->event.id];
                }
                if (it->is_upper_bound()) {
                    for (b_it = all_examples.begin(); b_it != all_examples.end(); b_it++) {
                        b_it->g_bar += it->get_c(cp, cn) * q_sva[b_it->event.id];
                    }
                }
            }
        }

        int iter = 0;
        int counter = min(n_examples, 1000) + 1;
        SolutionVector *sva;
        SolutionVector *svb;

        while (true) {
            if (--counter == 0) {
                counter = min(n_examples, 1000);
                if (shrinking) {
                    do_shrinking();
                }
            }

            SolutionVectorPair pair = select_working_pair();
            if (pair.is_optimal) {
                reconstruct_gradient();
                reset_active_set();

                pair = select_working_pair();
                if (pair.is_optimal) {
                    break;
                } else {
                    counter = 1;
                }
            }

            sva = pair.sva;
            svb = pair.svb;

            ++iter;

            if (iter > MAXITER) {
                cout << "Solver reached maximum iterations, aborting" << endl;
                break;
            }

            q->get_q(*sva, active, q_sva);
            q->get_q(*svb, active, q_svb);

            double c_i = sva->get_c(cp, cn);
            double c_j = svb->get_c(cp, cn);

            double old_alpha_i = sva->alpha;
            double old_alpha_j = svb->alpha;

            double quad_coef = q->eval_diagonal(*sva) + q->eval_diagonal(*svb)
                + 2.0 * q_sva[svb->event.id];
            if (quad_coef <= 0) {
                quad_coef = 1e-12;
            }

            if (sva->event.oid != svb->event.oid) {
                double delta = (-sva->g - svb->g) / quad_coef;
                double diff = sva->alpha - svb->alpha;
                sva->alpha += delta;
                svb->alpha += delta;

                if (diff > 0) {
                    if (svb->alpha < 0) {
                        svb->alpha = 0;
                        sva->alpha = diff;
                    }
                } else {
                    if (sva->alpha < 0) {
                        sva->alpha = 0;
                        svb->alpha = -diff;
                    }
                }

                if (diff > c_i - c_j) {
                    if (sva->alpha > c_i) {
                        sva->alpha = c_i;
                        svb->alpha = c_i - diff;
                    }
                } else {
                    if (svb->alpha > c_j) {
                        svb->alpha = c_j;
                        sva->alpha = c_j + diff;
                    }
                }
            } else {
                double delta = (sva->g - svb->g) / quad_coef;
                double sum = sva->alpha + svb->alpha;
                sva->alpha -= delta;
                svb->alpha += delta;

                if (sum > c_i) {
                    if (sva->alpha > c_i) {
                        sva->alpha = c_i;
                        svb->alpha = sum - c_i;
                    }
                } else {
                    if (svb->alpha < 0) {
                        svb->alpha = 0;
                        sva->alpha = sum;
                    }
                }

                if (sum > c_j) {
                    if (svb->alpha > c_j) {
                        svb->alpha = c_j;
                        svb->alpha = sum - c_j;
                    }
                } else {
                    if (sva->alpha < 0) {
                        sva->alpha = 0;
                        svb->alpha = sum;
                    }
                }
            }

            double delta_alpha_i = sva->alpha - old_alpha_i;
            double delta_alpha_j = svb->alpha - old_alpha_j;

            if (delta_alpha_i == 0 && delta_alpha_j == 0) {
                cout << "Pair is optimal within available numeric precision, but this is still larger than requested eps = "
                    << eps << "." << endl;
                break;
            }

            for (it = active.begin(); it != active.end(); it++) {
                int i = it - active.begin();
                it->g += q_sva[i] * delta_alpha_i + q_svb[i] * delta_alpha_j;
            }

            bool ui = sva->is_upper_bound();
            bool uj = svb->is_lower_bound();
            sva->update_alpha_status(cp, cn);
            svb->update_alpha_status(cp, cn);

            if (ui != sva->is_upper_bound()) {
                q->get_q(*sva, active, inactive, q_all);
                if (ui) {
                    for (it = all_examples.begin(); it != all_examples.end(); it++) {
                        it->g_bar -= c_i * q_all[it->event.id];
                    }
                } else {
                    for (it = all_examples.begin(); it != all_examples.end(); it++) {
                        it->g_bar += c_i * q_all[it->event.id];
                    }
                }
            }

            if (uj != svb->is_upper_bound()) {
                q->get_q(*svb, active, inactive, q_all);
                if (uj) {
                    for (it = all_examples.begin(); it != all_examples.end(); it++) {
                        it->g_bar -= c_j * q_all[it->event.id];
                    }
                } else {
                    for (it = all_examples.begin(); it != all_examples.end(); it++) {
                        it->g_bar += c_j * q_all[it->event.id];
                    }
                }
            }
        }

        cout << "Optimization finished after " << iter << " iterations." << endl;
        return iter;
    }

    void Solver::do_shrinking() {
        int i;
        double g_max1 = numeric_limits<double>::min();
        double g_max2 = g_max1;

        vector<SolutionVector>::iterator it;
        for (it = active.begin(); it != active.end(); it++) {
            Event &e = it->event;
            if (e.oid == 1) {
                if (!it->is_upper_bound()) {
                    if (-it->g >= g_max1) {
                        g_max1 = -it->g;
                    }
                }
                if (!it->is_lower_bound()) {
                    if (it->g >= g_max2) {
                        g_max2 = it->g;
                    }
                }
            } else {
                if (!it->is_upper_bound()) {
                    if (-it->g >= g_max2) {
                        g_max2 = -it->g;
                    }
                }
                if (!it->is_lower_bound()) {
                    if (it->g >= g_max1) {
                        g_max1 = it->g;
                    }
                }
            }
        }

        if (!unshrink && g_max1 + g_max2 <= eps * 10) {
            unshrink = true;
            reconstruct_gradient();
            reset_active_set();
        }

        vector<SolutionVector> inactive_list;
        for (it = active.begin(); it != active.end();) {
            if (it->is_shrinkable(g_max1, g_max2)) {
                inactive_list.push_back(*it);
                it = active.erase(it);
            } else {
                it++;
            }
        }

        int n_active = active.size();
        q_sva.resize(n_active, 0.0);
        q_svb.resize(n_active, 0.0);

        q->maintain_cache(active, inactive_list);

        int n_inactive = inactive_list.size();
        inactive_list.resize(n_inactive + inactive.size());
        copy(inactive.begin(), inactive.end(), inactive_list.begin() + n_inactive);

        sort(active.begin(), active.end(), cmp_solution_vector);
        sort(inactive.begin(), inactive.end(), cmp_solution_vector);
    }

    void Solver::reconstruct_gradient() {
        int n_active = active.size();
        if (n_active == n_examples) {
            return;
        }

        int nr_free = 0;
        vector<SolutionVector>::iterator it, itb;
        for (it = inactive.begin(); it != inactive.end(); it++) {
            it->g = it->g_bar + it->linear_term;
        }

        for (it = active.begin(); it != active.end(); it++) {
            if (it->is_free()) {
                nr_free++;
            }
        }

        if (nr_free * n_examples > 2 * n_active * (n_examples - n_active)) {
            for (it = inactive.begin(); it != inactive.end(); it++) {
                q->get_q(*it, active, q_sva);
                for (itb = active.begin(); itb != active.end(); itb++) {
                    if (itb->is_free()) {
                        itb->g += it->alpha * q_sva[itb->event.id];
                    }
                }
            }
        } else {
            for (it = active.begin(); it != active.end(); it++) {
                if (it->is_free()) {
                    q->get_q(*it, active, inactive, q_all);
                    for (itb = inactive.begin(); itb != inactive.end(); itb++) {
                        itb->g += it->alpha * q_all[itb->event.id];
                    }
                }
            }
        }
    }

    void Solver::reset_active_set() {
        active = all_examples;
        sort(active.begin(), active.end(), cmp_solution_vector);
        inactive.clear();
        q_sva.resize(n_examples);
        q_svb.resize(n_examples);
    }

    SolutionVectorPair Solver::select_working_pair() {
        double gmax = numeric_limits<double>::min();
        double gmax2 = gmax;
        SolutionVector *gmax_sv = NULL;
        SolutionVector *gmin_sv = NULL;
        double obj_diff_min = numeric_limits<double>::max();

        vector<SolutionVector>::iterator it;
        for (it = active.begin(); it != active.end(); it++) {
            if (it->event.oid == 1) {
                if (!it->is_upper_bound()) {
                    if (-it->g >= gmax) {
                        gmax = -it->g;
                        gmax_sv = &(*it);
                    }
                }
            } else {
                if (!it->is_lower_bound()) {
                    if (it->g >= gmax) {
                        gmax = it->g;
                        gmax_sv = &(*it);
                    }
                }
            }
        }

        if (gmax_sv != NULL) {
            q->get_q(*gmax_sv, active, q_sva);
        }

        for (it = active.begin(); it != active.end(); it++) {
            if (it->event.oid == 1) {
                if (!it->is_lower_bound()) {
                    double grad_diff = gmax + it->g;
                    if (it->g >= gmax2) {
                        gmax2 = it->g;
                    }
                    if (grad_diff > 0) {
                        double obj_diff;
                        double quad_coef = q->eval_diagonal(*gmax_sv) + q->eval_diagonal(*it)
                            - 2.0 * (gmax_sv->event.oid == 1 ? 1.0 : -1.0)
                            * q_sva[it->event.id];

                        if (quad_coef > 0) {
                            obj_diff = -(grad_diff * grad_diff) / quad_coef;
                        } else {
                            obj_diff = -(grad_diff * grad_diff) / 1e-12f;
                        }

                        if (obj_diff <= obj_diff_min) {
                            gmin_sv = &(*it);
                            obj_diff_min = obj_diff;
                        }
                    }
                }
            } else {
                if (!it->is_upper_bound()) {
                    double grad_diff = gmax - it->g;
                    if (-it->g >= gmax2) {
                        gmax2 = -it->g;
                    }

                    if (grad_diff > 0) {
                        double obj_diff;
                        double quad_coef = q->eval_diagonal(*gmax_sv) + q->eval_diagonal(*it)
                            - 2.0 * (gmax_sv->event.oid == 1 ? 1.0 : -1.0)
                            * q_sva[it->event.id];

                        if (quad_coef > 0) {
                            obj_diff = -(grad_diff * grad_diff) / quad_coef;
                        } else {
                            obj_diff = -(grad_diff * grad_diff) / 1e-12f;
                        }

                        if (obj_diff <= obj_diff_min) {
                            gmin_sv = &(*it);
                            obj_diff_min = obj_diff;
                        }
                    }
                }
            }
        }

        SolutionVectorPair pair(gmax_sv, gmin_sv, gmax+gmax2<eps);
        return pair;
    }
}

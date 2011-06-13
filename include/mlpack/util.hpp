/*
 * =====================================================================================
 *
 *       Filename:  util.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  13/06/11 11:38:00
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#ifndef MLPACK_UTIL_H
#define MLPACK_UTIL_H

#include <mlpack/events.hpp>

using namespace std;

namespace mlpack {
    enum AlphaStatus { LOWER_BOUND, UPPER_BOUND, FREE };

    struct SolutionVector {
        Event event;
        double linear_term;
        double alpha;
        double g_bar;
        double g;
        AlphaStatus alpha_status;

        SolutionVector() {}

        SolutionVector(Event e, double lt) {
            event = e;
            linear_term = lt;
        }

        SolutionVector(Event e, double lt, double a) {
            event = e;
            linear_term = lt;
            alpha = a;
        }

        void update_alpha_status(double cp, double cn) {
            if (alpha >= get_c(cp, cn)) {
                alpha_status = UPPER_BOUND;
            } else if (alpha <= 0) {
                alpha_status = LOWER_BOUND;
            } else {
                alpha_status = FREE;
            }
        }

        bool is_free() {
            return alpha_status == FREE;
        }

        bool is_shrinkable(double g_max1, double g_max2) {
            if (is_upper_bound()) {
                if (event.oid > 0) {
                    return -g > g_max1;
                } else {
                    return -g > g_max2;
                }
            } else if (is_lower_bound()) {
                if (event.oid > 0) {
                    return g > g_max2;
                } else {
                    return g > g_max1;
                }
            } else {
                return false;
            }
        }

        bool is_shrinkable(double g_max1, double g_max2, double g_max3, double g_max4) {
            if (is_upper_bound()) {
                if (event.oid > 0) {
                    return -g > g_max1;
                } else {
                    return -g > g_max4;
                }
            } else if (is_lower_bound()) {
                if (event.oid > 0) {
                    return g > g_max2;
                } else {
                    return g > g_max3;
                }
            } else {
                return false;
            }
        }

        bool is_upper_bound() {
            return alpha_status == UPPER_BOUND;
        }

        bool is_lower_bound() {
            return alpha_status == LOWER_BOUND;
        }

        double get_c(double cp, double cn) {
            return event.oid > 0 ? cp : cn;
        }
    };

    struct SolutionVectorPair {
        SolutionVector *sva;
        SolutionVector *svb;
        bool is_optimal;

        SolutionVectorPair(SolutionVector *_sva, SolutionVector *_svb, bool _optimal) {
            sva = _sva;
            svb = _svb;
            is_optimal = _optimal;
        }
    };

    bool cmp_solution_vector(const SolutionVector &sva, const SolutionVector &svb);
}

#endif

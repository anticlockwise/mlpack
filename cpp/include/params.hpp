/*
 * =====================================================================================
 *
 *       Filename:  params.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  31/05/11 12:44:02
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#ifndef PARAMS_H
#define PARAMS_H

#include <vector>

using namespace std;

struct Parameters {
    vector<double> params;
    vector<int> outcomes;

    Parameters() {}

    Parameters(vector<double> ps, vector<int> os) {
        params = ps;
        outcomes = os;
    }

    void set(int oi, double param) {
        params[oi] = param;
    }

    void update(int oi, double param) {
        params[oi] += param;
    }
};

class MaxentParameters {
    public:
        vector<Parameters> params;
        int n_outcomes;
        double corr_constant;
        double const_inverse;
        double corr_param;

        MaxentParameters(vector<Parameters> ps, double corr_p,
                double corr_const, int n_out) {
            params = ps;
            corr_param = corr_p;
            corr_constant = corr_const;
            const_inverse = 1.0 / corr_constant;
            n_outcomes = n_out;
        }
};

#endif

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

#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <vector>

using namespace std;

namespace mlpack {
    class Parameters {
        private:
            friend class boost::serialization::access;
            template <class Archive>
                void serialize(Archive &ar, const unsigned int version) {
                    ar & params;
                    ar & outcomes;
                }

        public:
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

            void print() {
                size_t n = params.size();
                int i;
                cout << n << ": ";
                for (i = 0; i < n; i++) {
                    cout << " " << params[i];
                }
                cout << endl;
            }
    };

    class MaxentParameters {
        private:
            friend class boost::serialization::access;
            template <class Archive>
                void serialize(Archive &ar, const unsigned int version) {
                    ar & params;
                    ar & n_outcomes;
                    ar & corr_constant;
                    ar & const_inverse;
                    ar & corr_param;
                }

        public:
            vector<Parameters> params;
            int n_outcomes;
            double corr_constant;
            double const_inverse;
            double corr_param;

            MaxentParameters() {}

            MaxentParameters(vector<Parameters> ps, double corr_p,
                    double corr_const, int n_out) {
                params = ps;
                corr_param = corr_p;
                corr_constant = corr_const;
                const_inverse = 1.0 / corr_constant;
                n_outcomes = n_out;
            }
    };
}

#endif

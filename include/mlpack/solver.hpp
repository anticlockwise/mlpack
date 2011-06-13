/*
 * =====================================================================================
 *
 *       Filename:  solver.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  13/06/11 11:01:10
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#ifndef MLPACK_SOLVER_H
#define MLPACK_SOLVER_H

#include <mlpack/model.hpp>
#include <mlpack/events.hpp>
#include <mlpack/util.hpp>
#include <mlpack/qmatrix.hpp>
#include <vector>
#include <map>
#include <limits>
#include <algorithm>

using namespace std;

namespace mlpack {
    const int MAXITER = 50000;

    class SolutionModel {
        public:
            int svm_type;
    };

    struct AlphaModel {
        map<FeatureSet, double> sv_map;
        vector<FeatureSet> sv_list;
        int n_svs;
        vector<double> alphas;
        double rho;
        double obj;
        double upper_bound_positive;
        double upper_bound_negative;
    };

    class Solver {
        shared_ptr<QMatrix> q;
        vector<double> q_sva;
        vector<double> q_svb;
        vector<double> q_all;

        double eps;
        bool unshrink;
        bool shrinking;

        protected:
        vector<SolutionVector> all_examples;
        vector<SolutionVector> active;
        vector<SolutionVector> inactive;
        double cp, cn;
        int n_examples;

        void calculate_rho(AlphaModel &model);

        int optimize();

        void init_active_set();

        void do_shrinking();

        void reconstruct_gradient();

        void reset_active_set();

        SolutionVectorPair select_working_pair();

        public:
        Solver(vector<SolutionVector> sol_vecs, shared_ptr<QMatrix> qm,
                double _cp, double _cn, double _eps, bool _shrinking) {
            q = qm;
            cp = _cp;
            cn = _cn;
            eps = _eps;
            shrinking = _shrinking;

            all_examples = sol_vecs;

            n_examples = all_examples.size();
            q_all.resize(n_examples, 0.0);
        }

        virtual void solve(SVMModel &model);
    };

    class BinarySolver : public Solver {
        public:
            void solve(SVMModel &model);
    };
}

#endif

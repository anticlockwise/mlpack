/*
 * =====================================================================================
 *
 *       Filename:  qmatrix.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  13/06/11 10:49:00
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#ifndef MLPACK_QMATRIX_H
#define MLPACK_QMATRIX_H

#include <mlpack/events.hpp>
#include <mlpack/kernel.hpp>
#include <mlpack/util.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <algorithm>
#include <cfloat>
#include <cmath>

using namespace std;
using boost::shared_ptr;

typedef vector<vector<double> > Matrix;

double const NOT_CACHED = DBL_MAX;

namespace mlpack {
    class QMatrix;

    class KernelQMatrix;

    class RecentActivitySquareCache {

        Matrix data;
        vector<double> diagonal;
        int max_cached_rank;

        long hits;
        long misses;
        long widemisses;
        long diagonalhits;
        long diagonalmisses;

        public:
        RecentActivitySquareCache(int num_examples, int cache_rows) {
            hits = 0;
            misses = 0;
            widemisses = 0;
            diagonalhits = 0;
            diagonalmisses = 0;

            max_cached_rank = min(num_examples, cache_rows);
            data.reserve(max_cached_rank);
            int i;
            for (i = 0; i < max_cached_rank; i++) {
                data[i].resize(max_cached_rank, NOT_CACHED);
            }

            diagonal.resize(num_examples, NOT_CACHED);
        }

        double get(SolutionVector &a, SolutionVector &b, KernelQMatrix *q);

        double get_diagonal(SolutionVector &a, KernelQMatrix *q);

        void get(SolutionVector &a, vector<SolutionVector> &active, vector<double> &buf,
                KernelQMatrix *q);

        void get(SolutionVector &a, vector<SolutionVector> &active,
                vector<SolutionVector> &inactive, vector<double> &buf,
                KernelQMatrix *q);

        void maintain_cache(vector<SolutionVector> &active,
                vector<SolutionVector> &inactive);

        void swap_by_solution_vector(SolutionVector &a, SolutionVector &b);

        void swap_by_rank(int rank_a, int rank_b);
    };

    class QMatrix {
        public:
            virtual double eval_diagonal(SolutionVector &a) = 0;

            virtual void get_q(SolutionVector &sva, vector<SolutionVector> &active, vector<double> &buf) = 0;

            virtual void get_q(SolutionVector &sva, vector<SolutionVector> &active,
                    vector<SolutionVector> &inactive, vector<double> &buf) = 0;

            virtual void init_ranks(vector<SolutionVector> &all_examples) = 0;

            virtual void maintain_cache(vector<SolutionVector> &active,
                    vector<SolutionVector> &inactive) = 0;

            virtual string perf_string() = 0;
    };

    class KernelQMatrix : public QMatrix {
        protected:
            shared_ptr<Kernel> kernel;

            shared_ptr<RecentActivitySquareCache> cache;

        public:
            KernelQMatrix(shared_ptr<Kernel> k, int num_examples,
                    int cache_rows) {
                kernel = k;
                cache = shared_ptr<RecentActivitySquareCache>(new
                        RecentActivitySquareCache(num_examples, cache_rows));
            }

            double eval_diagonal(SolutionVector &a);

            void get_q(SolutionVector &, vector<SolutionVector> &, vector<double> &);

            void get_q(SolutionVector &, vector<SolutionVector> &, vector<SolutionVector> &,
                    vector<double> &);

            void init_ranks(vector<SolutionVector> &);

            void maintain_cache(vector<SolutionVector> &, vector<SolutionVector> &);

            string perf_string();

            double evaluate(SolutionVector &a, SolutionVector &b);

            virtual double compute_q(SolutionVector &a, SolutionVector &b) = 0;
    };

    class BasicKernelQMatrix : public KernelQMatrix {
        public:
            BasicKernelQMatrix(shared_ptr<Kernel> kernel, int n_examples, int max_rank)
                : KernelQMatrix(kernel, n_examples, max_rank) {}

            double compute_q(SolutionVector &a, SolutionVector &b);
    };

    class BinaryInvertingKernelQMatrix : public BasicKernelQMatrix {
        public:
            BinaryInvertingKernelQMatrix(shared_ptr<Kernel> kernel,
                    int n_examples, int max_rank)
                : BasicKernelQMatrix(kernel, n_examples, max_rank) {}

            double compute_q(SolutionVector &a, SolutionVector &b);
    };
}

#endif

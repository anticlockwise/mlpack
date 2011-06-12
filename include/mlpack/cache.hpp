/*
 * =====================================================================================
 *
 *       Filename:  cache.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  06/11/2011 22:52:53
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#ifndef MLPACK_CACHE_H
#define MLPACK_CACHE_H

#include <mlpack/svm.hpp>

#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

using namespace std;

typedef vector<vector<double> > Matrix;

namespace mlpack {
    class RecentActivitySquareCache {
        double const NOT_CACHED = numeric_limits<double>::max();

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
            for (i = 0; i < max_cached_rank) {
                data[i].resize(max_cached_rank, NOT_CACHED);
            }

            diagonal.resize(num_examples, NOT_CACHED);
        }

        double get(SolutionVector &a, SolutionVector &b, KernelQMatrix *q) {
            Event &ea = a.event;
            Event &eb = b.event;

            if (ea.id == eb.id)
                return get_diagonal(a, b);

            if (ea.id >= max_cached_rank || eb.id >= max_cached_rank) {
                widemisses++;
                return q->compute_q(a, b);
            }

            double result = data[ea.id][eb.id];
            if (result == NOT_CACHED) {
                result = q->compute_q(a, b);
                data[ea.id][eb.id] = result;
                data[eb.id][ea.id] = result;
                misses++;
            } else {
                hits++;
            }
            return result;
        }

        double get_diagonal(SolutionVector &a, SolutionVector &b,
                KernelQMatrix *q) {
            Event &ea = a.event;
            Event &eb = b.event;
            double result = diagonal[ea.id];
            if (result == NOT_CACHED) {
                result = q->compute_q(a, b);
                diagonal[ea.id] = result;
                diagonalmisses++;
            } else {
                diagonalhits++;
            }
            return result;
        }

        void get(SolutionVector &a, vector<SolutionVector> &active, vector<double> &buf,
                KernelQMatrix *q) {
            Event &ea;
            int i;
            int as = active.size();
            if (ea.id >= max_cached_rank) {
                for (i = 0; i < as; i++) {
                    buf[i] = q->compute_q(a, active[i]);
                    widemisses++;
                }
                return;
            }

            vector<double> &row = data[ea.id];
            int cached_n_active = min(row.size(), as);

            for (i = 0; i < cached_n_active; i++) {
                SolutionVector &b = active[i];
                if (row[i] == NOT_CACHED) {
                    row[i] = q->compute_q(a, b);
                    data[b.event.id][ea.id] = row[i];
                    misses++;
                } else {
                    hits++;
                }
            }

            copy(row.begin(), row.end(), buf.begin());
            for (i = cached_n_active; i < as; i++) {
                SolutionVector &b = active[i];
                buf[i] = q->compute_q(a, b);
                widemisses++;
            }
        }

        void get(SolutionVector &a, vector<SolutionVector> &active,
                vector<SolutionVector> &inactive, vector<double> &buf,
                KernelQMatrix *q) {
            get(a, active, buf);

            Event &ea = a.event;
            if (ea.id >= max_cached_rank) {
                int i = active.size();
                vector<SolutionVector>::iterator it;
                for (it = inactive.begin(); it != inactive.end(); it++) {
                    buf[i] = q->compute(a, *it);
                    widemisses++;
                    i++;
                }
            } else {
                vector<double> &row = data[ea.id];
                int i = active.size();
                vector<SolutionVector>::iterator it;
                for (it = inactive.begin(); it != inactive.end(); it++) {
                    SolutionVector &b = *it;
                    Event &eb = b.event;
                    if (eb.id >= max_cached_rank) {
                        buf[i] = q->compute(a, b);
                        widemisses++;
                    } else {
                        if (row[eb.id] == NOT_CACHED) {
                            row[eb.id] = q->compute(a, b);
                            data[eb.id][ea.id] = row[eb.id];
                            misses++;
                        } else {
                            hits++;
                        }
                        buf[i] = row[eb.id];
                    }
                    i++;
                }
            }
        }

        void maintain_cache(vector<SolutionVector> &active,
                vector<SolutionVector> &inactive) {
            int part_rank = active.size();
            int in_size = inactive.size();

            int i = 0;
            int j = 0;

            while (true) {
                while (i < part_rank && active[i].event.id < part_rank) {
                    i++;
                }

                while (j < in_size && inactive[j].event.id >= part_rank) {
                    j++;
                }

                if (i < part_rank && j < in_size) {
                    swap_by_solution_vector(active[i], inactive[i]);
                    i++;
                    j++;
                } else {
                    break;
                }
            }
        }
    };
}

#endif

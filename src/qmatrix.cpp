/*
 * =====================================================================================
 *
 *       Filename:  qmatrix.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  13/06/11 10:50:40
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include <mlpack/qmatrix.hpp>

namespace mlpack {
    double RecentActivitySquareCache::get(SolutionVector &a, SolutionVector &b, KernelQMatrix *q) {
        Event &ea = a.event;
        Event &eb = b.event;

        if (ea.id == eb.id)
            return get_diagonal(a, q);

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

    double RecentActivitySquareCache::get_diagonal(SolutionVector &a,
            KernelQMatrix *q) {
        Event &ea = a.event;
        double result = diagonal[ea.id];
        if (result == NOT_CACHED) {
            result = q->compute_q(a, a);
            diagonal[ea.id] = result;
            diagonalmisses++;
        } else {
            diagonalhits++;
        }
        return result;
    }

    void RecentActivitySquareCache::get(SolutionVector &a, vector<SolutionVector> &active, vector<double> &buf,
            KernelQMatrix *q) {
        Event &ea = a.event;
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
        int cached_n_active = min((int)row.size(), as);

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

    void RecentActivitySquareCache::get(SolutionVector &a, vector<SolutionVector> &active,
            vector<SolutionVector> &inactive, vector<double> &buf,
            KernelQMatrix *q) {
        get(a, active, buf, q);

        Event &ea = a.event;
        if (ea.id >= max_cached_rank) {
            int i = active.size();
            vector<SolutionVector>::iterator it;
            for (it = inactive.begin(); it != inactive.end(); it++) {
                buf[i] = q->compute_q(a, *it);
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
                    buf[i] = q->compute_q(a, b);
                    widemisses++;
                } else {
                    if (row[eb.id] == NOT_CACHED) {
                        row[eb.id] = q->compute_q(a, b);
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

    void RecentActivitySquareCache::maintain_cache(vector<SolutionVector> &active,
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

    void RecentActivitySquareCache::swap_by_solution_vector(SolutionVector &a, SolutionVector &b) {
        swap_by_rank(a.event.id, b.event.id);
        int tmp = a.event.id;
        a.event.id = b.event.id;
        b.event.id = tmp;
    }

    void RecentActivitySquareCache::swap_by_rank(int rank_a, int rank_b) {
        double tmp = diagonal[rank_a];
        diagonal[rank_a] = diagonal[rank_b];
        diagonal[rank_b] = tmp;

        if (rank_a >= max_cached_rank && rank_b >= max_cached_rank) {

        } else if (rank_a < max_cached_rank && rank_b < max_cached_rank) {
            vector<double> dtmp = data[rank_a];
            data[rank_a] = data[rank_b];
            data[rank_b] = dtmp;

            int s = data.size();
            int i;
            for (i = 0; i < s; i++) {
                vector<double> &drow = data[i];
                double d = drow[rank_a];
                drow[rank_a] = drow[rank_b];
                drow[rank_b] = d;
            }
        } else if (rank_a < max_cached_rank) {
            fill(data[rank_a].begin(), data[rank_b].end(), NOT_CACHED);
        } else {
            fill(data[rank_b].begin(), data[rank_b].end(), NOT_CACHED);
        }
    }

    double KernelQMatrix::eval_diagonal(SolutionVector &a) {
        return cache->get_diagonal(a, this);
    }

    void KernelQMatrix::get_q(SolutionVector &a, vector<SolutionVector> &active,
            vector<double> &buf) {
        cache->get(a, active, buf, this);
    }

    void KernelQMatrix::get_q(SolutionVector &a, vector<SolutionVector> &active,
            vector<SolutionVector> &inactive, vector<double> &buf) {
        cache->get(a, active, inactive, buf, this);
    }

    void KernelQMatrix::init_ranks(vector<SolutionVector> &all_examples) {
    }

    void KernelQMatrix::maintain_cache(vector<SolutionVector> &active, vector<SolutionVector> &inactive) {
        cache->maintain_cache(active, inactive);
    }

    string KernelQMatrix::perf_string() {
        return "";
    }

    double BasicKernelQMatrix::compute_q(SolutionVector &a, SolutionVector &b) {
        Event &ea = a.event;
        Event &eb = b.event;
        return kernel->eval(ea.context, eb.context);
    }

    double BinaryInvertingKernelQMatrix::compute_q(SolutionVector &a, SolutionVector &b) {
        Event &ea = a.event;
        Event &eb = b.event;
        return ((ea.oid == eb.oid) ? 1.0 : -1.0) * kernel->eval(ea.context, eb.context);
    }
}

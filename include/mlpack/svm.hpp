/*
 * =====================================================================================
 *
 *       Filename:  svm.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  03/06/11 16:31:29
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include <mlpack/model.hpp>
#include <mlpack/trainer.hpp>
#include <mlpack/index.hpp>
#include <mlpack/feature.hpp>
#include <mlpack/events.hpp>
#include <mlpack/kernel.hpp>
#include <mlpack/cache.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;
using boost::property_tree::ptree;
using boost::shared_ptr;

namespace mlpack {
    enum KernelType { LINEAR = 0, POLYNOMIAL = 1, RADIAL = 2, SIGMOID = 3 };

    enum SVMType { C_SVC = 0, NU_SVC = 1, ONE_CLASS = 2, EPSILON_SVR = 3, NU_SVR = 4 };

    enum AlphaStatus { LOWER_BOUND, UPPER_BOUND, FREE };

    struct SolutionVector {
        Event event;
        double linear_term;
        double alpha;
        double g_bar;
        double g;
        AlphaStatus alpha_status;

        SolutionVector(Event e, double lt) {
            event = e;
            linear_term = lt;
        }

        SolutionVector(Event e, double lt, double a) {
            event = e;
            linear_term = lt;
            alpha = a;
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

    class SVMParameters {
        public:
            int svm_type;
            int kernel_type;
            double degree;
            double gamma;
            double coef0;

            double cache_size;
            double eps;
            double c;
            int nr_weight;
            vector<int> weight_label;
            vector<double> weight;
            double nu;
            double p;
            int shrinking;
            bool probability;
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

    class Solver {
    };

    class SVMModel : public BaseModel {
        public:
            shared_ptr<SVMParameters> params;
            int nr_class;
            int l;

            vector<double> eval(FeatureSet context);

            SVMModel() {}

            virtual ~SVMModel() {}
    };

    class BinarySVMTrainer : public Trainer<SVMModel> {
        private:
            shared_ptr<Kernel> kernel;

            shared_ptr<SVMParameters> params;

            void init_params(ptree &c) {
                params = shared_ptr<SVMParameters>(new SVMParameters);
                params->c = c.get<double>("svm.C", 1.0);
                params->kernel_type = c.get<int>("svm.kernel_type", RADIAL);
                params->degree = c.get<double>("svm.degree", 3.0);
                params->gamma = c.get<double>("svm.gamma", 0.0);
                params->coef0 = c.get<double>("svm.coef0", 0.0);
                params->nu    = c.get<double>("svm.nu", 0.5);
                params->cache_size = c.get<int>("svm.cache_size", 100);
                params->eps = c.get<double>("svm.eps", 1e-3);
                params->p = c.get<double>("svm.p", 0.1);
                params->shrinking = c.get<int>("svm.shrinking", 1);
                params->probability = c.get<bool>("probability", false);
                params->nr_weight = c.get<int>("svm.nr_weight", 0);
                // TODO: add weighting configuration to C -> weighted_c[i] = C * weight[j]
            }

            void train_scaled(SVMModel &model, DataIndexer &di, ptree config);
            void train_single_class(SVMModel &model, DataIndexer &di,
                    double weight_cp, double weight_cn);

        public:
            SVMModel train(DataIndexer &di, ptree config);

            void set_heldout_data(EventSpace events);
    };
}

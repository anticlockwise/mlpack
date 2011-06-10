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

    struct DecisionFunction {
        vector<double> alpha;
        double rho;
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
            int probability;
    };

    class QMatrix {
        protected:
            shared_ptr<Kernel> kernel;
            EventSpace contexts;

        public:
            QMatrix(EventSpace &events, shared_ptr<SVMParameters> params) {
                switch (params->kernel_type) {
                    case LINEAR:
                        kernel = shared_ptr<Kernel>(new LinearKernel);
                        break;
                    case POLYNOMIAL:
                        kernel = shared_ptr<Kernel>(new PolynomialKernel(
                                    params->gamma, params->coef0, params->degree));
                        break;
                    case RADIAL:
                        kernel = shared_ptr<Kernel>(new RadialKernel(
                                    params->gamma));
                        break;
                    case SIGMOID:
                        kernel = shared_ptr<Kernel>(new SigmoidNeuralKernel(
                                    params->gamma, params->coef0));
                        break;
                }
            }

            virtual vector<double> get_q(int i , int l) = 0;

            virtual vector<double> get_qd() = 0;
    };

    class SVCQMatrix : public QMatrix {
        private:
            vector<double> qd;

        public:
            SVCQMatrix(EventSpace events, shared_ptr<SVMParameters> params);

            vector<double> get_qd();

            vector<double> get_q(int i, int l);
    };

    class Solver {
        private:
            enum { LOWER_BOUND = 0, UPPER_BOUND = 1, FREE = 2};
            vector<double> gradient;
            vector<double> g_bar;
            vector<int> alpha_status;
            vector<int> active_set;
            double cp;
            double cn;

        public:
            void solve(int ne, shared_ptr<QMatrix> q,
                    vector<double> p, EventSpace &events,
                    vector<double> a, double c_p, double c_n,
                    double eps, int shrinking);

            double get_c(int o) {
                return o > 0 ? cp : cn;
            }

            bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
            bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
            bool is_free(int i) { return alpha_status[i] == FREE; }

            void update_alpha_status(double alpha, int i, int o) {
                if (alpha > get_c(o)) {
                    alpha_status[i] = UPPER_BOUND;
                } else if (alpha <= 0) {
                    alpha_status[i] = LOWER_BOUND;
                } else {
                    alpha_status[i] = FREE;
                }
            }
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

    class SVMTrainer : public Trainer<SVMModel> {
        private:
            shared_ptr<Kernel> kernel;

            shared_ptr<SVMParameters> learn_params;

            void init_params(shared_ptr<SVMParameters> params, ptree &c) {
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
                params->probability = c.get<int>("probability", 0);
                params->nr_weight = c.get<int>("svm.nr_weight", 0);
                // TODO: add weighting configuration to C -> weighted_c[i] = C * weight[j]
            }

            void group_classes(EventSpace &events, vector<int> &start, vector<int> &count, int n_classes);

            DecisionFunction train_single_class(EventSpace &subspace, shared_ptr<SVMParameters> params,
                    int wp, int wn);

        public:
            SVMModel train(DataIndexer &di, ptree config);

            void set_heldout_data(EventSpace events);
    };
}

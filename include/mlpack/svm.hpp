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
#include <boost/property_tree/ptree.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <cmath>

using namespace std;
using boost::property_tree::ptree;
using boost::shared_ptr;

namespace mlpack {
    const int LINEAR_KERNEL = 0;
    const int POLYNOMIAL_KERNEL = 1;
    const int RADIAL_KERNEL = 2;
    const int SIGMOID_KERNEL = 3;

    double cross_prod(shared_ptr<FeatureSet>, shared_ptr<FeatureSet>);

    class Kernel {
        public:
            virtual double eval(shared_ptr<FeatureSet>, shared_ptr<FeatureSet>) = 0;
    };

    class LinearKernel : public Kernel {
        public:
            double eval(shared_ptr<FeatureSet> fset1, shared_ptr<FeatureSet> fset2);
    };

    class PolynomialKernel : public Kernel {
        private:
            double lin_coef; // Linear coefficient a
            double const_coef; // Constant coefficient b
            double poly_degree; // Polynomial degree

        public:
            PolynomialKernel(double l, double c, double p) {
                lin_coef = l;
                const_coef = c;
                poly_degree = p;
            }

            double eval(shared_ptr<FeatureSet> fset1, shared_ptr<FeatureSet> fset2);
    };

    class RadialKernel : public Kernel {
        private:
            double rbf_gamma;

        public:
            RadialKernel(double r) {
                rbf_gamma = r;
            }

            double eval(shared_ptr<FeatureSet> fset1, shared_ptr<FeatureSet> fset2);
    };

    class SigmoidNeuralKernel : public Kernel {
        private:
            double lin_coef; // Linear coefficient a
            double const_coef; // Constant coefficient b

        public:
            SigmoidNeuralKernel(double l, double c) {
                lin_coef = l;
                const_coef = c;
            }

            double eval(shared_ptr<FeatureSet> fset1, shared_ptr<FeatureSet> fset2);
    };

    class SVMParameters {
        public:
            int type; // Classification/Regression
            double c; // upper bound C on alphas
            double eps; // Regression epsilon
            double cost_ratio; // Factor to multiply C for positive examples
            bool biased_hyperplane;
            bool shared_slack;
            long iterations;
            vector<double> alphas;
            vector<double> costs;
    };

    class SVMModel : public BaseModel {
        private:
            int kernel_type;
            vector<double> alpha;

        public:
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
                params->cost_ratio = c.get<double>("svm.cost_ratio", 1.0);
            }

        public:
            SVMModel train(DataIndexer &di, ptree config);

            void set_heldout_data(EventSpace events);
    };
}

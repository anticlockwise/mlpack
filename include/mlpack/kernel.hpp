/*
 * =====================================================================================
 *
 *       Filename:  kernel.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  10/06/11 12:10:39
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include <mlpack/feature.hpp>
#include <mlpack/events.hpp>
#include <vector>
#include <cmath>

using namespace std;

namespace mlpack {
    double cross_prod(FeatureSet&, FeatureSet&);

    class Kernel {
        public:
            virtual double eval(FeatureSet&, FeatureSet&) = 0;
    };

    class LinearKernel : public Kernel {
        public:
            double eval(FeatureSet &fset1, FeatureSet &fset2);
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

            double eval(FeatureSet &fset1, FeatureSet &fset2);
    };

    class RadialKernel : public Kernel {
        private:
            double rbf_gamma;

        public:
            RadialKernel(double r) {
                rbf_gamma = r;
            }

            double eval(FeatureSet &fset1, FeatureSet &fset2);
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

            double eval(FeatureSet &fset1, FeatureSet &fset2);
    };
}

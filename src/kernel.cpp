/*
 * =====================================================================================
 *
 *       Filename:  kernel.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  10/06/11 12:12:17
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include <mlpack/kernel.hpp>

namespace mlpack {
    double cross_prod(FeatureSet &fset1, FeatureSet &fset2) {
        double sum = 0.0;
        FeatureIterator fit;
        for (fit = fset1.begin(); fit != fset1.end(); fit++) {
            string name = fit->first;
            if (fset2.find(name) != fset2.end()) {
                Feature &f1 = fit->second;
                const Feature &f2 = fset2.get(name);
                sum += f1.value * f2.value;
            }
        }
        return sum;
    }

    double LinearKernel::eval(FeatureSet &fset1, FeatureSet &fset2) {
        return cross_prod(fset1, fset2);
    }

    double PolynomialKernel::eval(FeatureSet &fset1, FeatureSet &fset2) {
        double sum = cross_prod(fset1, fset2);
        return pow(lin_coef * sum + const_coef, poly_degree);
    }

    double RadialKernel::eval(FeatureSet &fset1, FeatureSet &fset2) {
        return exp(-rbf_gamma * (fset1.length_sq() - 2*cross_prod(fset1, fset2) + fset2.length_sq()));
    }

    double SigmoidNeuralKernel::eval(FeatureSet &fset1, FeatureSet &fset2) {
        double sum = cross_prod(fset1, fset2);
        return tanh(lin_coef * sum + const_coef);
    }
}

/*
 * =====================================================================================
 *
 *       Filename:  svm.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  03/06/11 16:35:09
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include <mlpack/svm.hpp>

namespace mlpack {
    double cross_prod(shared_ptr<FeatureSet> fset1, shared_ptr<FeatureSet> fset2) {
        double sum = 0.0;
        FeatureIterator fit;
        for (fit = fset1->begin(); fit != fset1->end(); fit++) {
            string name = fit->first;
            if (fset2->find(name) != fset2->end()) {
                Feature &f1 = fit->second;
                const Feature &f2 = fset2->get(name);
                sum += f1.value * f2.value;
            }
        }
        return sum;
    }

    double LinearKernel::eval(shared_ptr<FeatureSet> fset1, shared_ptr<FeatureSet> fset2) {
        return cross_prod(fset1, fset2);
    }

    double PolynomialKernel::eval(shared_ptr<FeatureSet> fset1, shared_ptr<FeatureSet> fset2) {
        double sum = cross_prod(fset1, fset2);
        return pow(lin_coef * sum + const_coef, poly_degree);
    }

    double RadialKernel::eval(shared_ptr<FeatureSet> fset1, shared_ptr<FeatureSet> fset2) {
        return exp(-rbf_gamma * (fset1->length_sq() - 2*cross_prod(fset1, fset2) + fset2->length_sq()));
    }

    double SigmoidNeuralKernel::eval(shared_ptr<FeatureSet> fset1, shared_ptr<FeatureSet> fset2) {
        double sum = cross_prod(fset1, fset2);
        return tanh(lin_coef * sum + const_coef);
    }

    vector<double> SVMModel::eval(FeatureSet context) {
        
    }

    SVMModel SVMTrainer::train(DataIndexer &di, ptree config) {
        EventSpace contexts = di.contexts();

        EventSpace::iterator cit;
        for (cit = contexts.begin(); cit != contexts.end(); cit++) {
            Event &e = (*cit);
            string outcome = e.outcome;
        }
    }

    void SVMTrainer::set_heldout_data(EventSpace events) {
        
    }
}

int main(int argc, char** argv) {
    return 0;
}

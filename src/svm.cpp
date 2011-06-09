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
        // 0 - Linear
        // 1 - Polynomial
        // 2 - Radial basis
        // 3 - SigmoidNeural
        int kernel_type = config.get<int>("svm.kernel", 0);
        double lin_coef, const_coef, poly_degree, rbf_gamma;
        switch (kernel_type) {
            case LINEAR_KERNEL:
                kernel = shared_ptr<Kernel>(new LinearKernel);
                break;
            case POLYNOMIAL_KERNEL:
                lin_coef = config.get<double>("svm.kernel.lin_coef", 0.1);
                const_coef = config.get<double>("svm.kernel.const_coef", 0.1);
                poly_degree = config.get<double>("svm.kernel.poly_degree", 1);
                kernel = shared_ptr<Kernel>(new PolynomialKernel(lin_coef, const_coef, poly_degree));
                break;
            case RADIAL_KERNEL:
                rbf_gamma = config.get<double>("svm.kernel.rbf_gammar", 0.1);
                kernel = shared_ptr<Kernel>(new RadialKernel(rbf_gamma));
                break;
            case SIGMOID_KERNEL:
                lin_coef = config.get<double>("svm.kernel.lin_coef", 0.1);
                const_coef = config.get<double>("svm.kernel.const_coef", 0.1);
                kernel = shared_ptr<Kernel>(new SigmoidNeuralKernel(lin_coef, const_coef));
                break;
            default:
                kernel = shared_ptr<Kernel>(new LinearKernel);
                break;
        }

        learn_params = shared_ptr<SVMParameters>(new SVMParameters);
        init_params(learn_params, config);

        EventSpace contexts = di.contexts();
        int n_events = di.num_events();
        vector<int> inconsistent(n_events);
        vector<int> unlabelled(n_events);
        vector<int> label(n_events);
        vector<double> alphas(n_events);
        learn_params->costs.reserve(n_events);
        int train_pos = 0;
        int train_neg = 0;
        int transduction = 0;

        EventSpace::iterator cit;
        for (cit = contexts.begin(); cit != contexts.end(); cit++) {
            Event &e = (*cit);
            int outcome = atoi(e.outcome.c_str());
            inconsistent[e.id] = 0;

            if (outcome == 0) {
                unlabelled[e.id] = 1;
                label[e.id] = 0;
                transduction = 1;
            } else if (outcome > 0) {
                learn_params->costs[e.id] = learn_params->c * learn_params->cost_ratio
                    * e.context.get_attr("cost_factor");
                label[e.id] = 1;
                train_pos++;
            } else if (outcome < 0) {
                learn_params->costs[e.id] = learn_params->c
                    * e.context.get_attr("cost_factor");
                label[e.id] = -1;
                train_neg++;
            } else {
                learn_params->costs[e.id] = 0;
            }
        }
    }

    void SVMTrainer::set_heldout_data(EventSpace events) {
        
    }
}

int main(int argc, char** argv) {
    return 0;
}

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

#ifndef MLPACK_SVM_H
#define MLPACK_SVM_H

#include <mlpack/model.hpp>
#include <mlpack/trainer.hpp>
#include <mlpack/index.hpp>
#include <mlpack/feature.hpp>
#include <mlpack/events.hpp>
#include <mlpack/kernel.hpp>
#include <mlpack/qmatrix.hpp>
#include <mlpack/solver.hpp>
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

                switch (params->kernel_type) {
                    case LINEAR:
                        kernel = shared_ptr<Kernel>(new LinearKernel);
                        break;
                    case POLYNOMIAL:
                        kernel = shared_ptr<Kernel>(new PolynomialKernel(params->gamma,
                                    params->coef0, params->degree));
                        break;
                    case RADIAL:
                        kernel = shared_ptr<Kernel>(new RadialKernel(params->gamma));
                        break;
                    case SIGMOID:
                        kernel = shared_ptr<Kernel>(new SigmoidNeuralKernel(params->gamma,
                                    params->coef0));
                        break;
                    default:
                        break;
                }
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

#endif

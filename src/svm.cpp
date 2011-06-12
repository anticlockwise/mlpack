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

    vector<double> SVMModel::eval(FeatureSet context) {
        
    }

    void BinarySVMTrainer::train_scaled(SVMModel &model, DataIndexer &di, ptree config) {
        init_params(config);

        double weight_cp = params->c;
        double weight_cn = params->c;

        train_single_class(model, di, weight_cp, weight_cn);
    }

    void BinarySVMTrainer::train_single_class(SVMModel &model, DataIndexer &di,
            double weight_cp, double weight_cn) {
        float linear_term = -1.0f;
        EventSpace events = di.contexts();
        int n_events = events.size();
        vector<SolutionVector> solution_vecs(n_events);

        EventSpace::iterator eit;
        for (eit = events.begin(); eit != events.end(); eit++) {
            SolutionVector sv(*eit, linear_term);
            solution_vecs.push_back(sv);
        }
    }

    SVMModel BinarySVMTrainer::train(DataIndexer &di, ptree config) {
        SVMModel model;
        train_scaled(model, di, config);
        return model;
    }

    void BinarySVMTrainer::set_heldout_data(EventSpace events) {
        
    }
}

int main(int argc, char** argv) {
    return 0;
}

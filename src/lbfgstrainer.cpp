/*
 * =====================================================================================
 *
 *       Filename:  lbfgstrainer.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  02/06/11 10:29:15
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include <mlpack/lbfgstrainer.hpp>

namespace mlpack {
    MaxentModel LBFGSTrainer::train(DataIndexer &di, ptree config) {
        MaxentModel model;
        return model;
    }

    void LBFGSTrainer::set_heldout_data(EventSpace events) {

    }
}

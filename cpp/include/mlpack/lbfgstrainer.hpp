/*
 * =====================================================================================
 *
 *       Filename:  lbfgstrainer.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  02/06/11 10:27:01
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include <boost/property_tree/ptree.hpp>
#include <mlpack/trainer.hpp>
#include <mlpack/index.hpp>
#include <mlpack/prior.hpp>
#include <mlpack/model.hpp>
#include <mlpack/events.hpp>
#include <vector>
#include <iostream>

using namespace std;
using boost::property_tree::ptree;

namespace mlpack {
    class LBFGSTrainer : public Trainer<MaxentModel> {
        public:
            MaxentModel train(DataIndexer &di, ptree config);

            void set_heldout_data(EventSpace events);
    };
}

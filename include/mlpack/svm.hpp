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

using namespace std;
using boost::property_tree::ptree;
using boost::shared_ptr;

namespace mlpack {
    class SVMModel : public BaseModel {
        public:
            vector<double> eval(FeatureSet context);

            SVMModel() {}

            virtual ~SVMModel() {}
    };

    class SVMTrainer : public Trainer<SVMModel> {
        public:
            SVMModel train(DataIndexer &di, ptree config);

            void set_heldout_data(EventSpace events);
    };
}

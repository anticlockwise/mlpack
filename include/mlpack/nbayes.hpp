/*
 * =====================================================================================
 *
 *       Filename:  nbayes.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  02/06/11 14:44:09
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#ifndef NBAYES_H
#define NBAYES_H

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/property_tree/ptree.hpp>
#include <vector>
#include <map>
#include <mlpack/events.hpp>
#include <mlpack/probs.hpp>
#include <mlpack/trainer.hpp>
#include <mlpack/index.hpp>
#include <mlpack/model.hpp>

using namespace std;
using boost::property_tree::ptree;
using boost::shared_ptr;

namespace mlpack {
    class NaiveBayesModel : public BaseModel {
        vector<double> priors;
        string dist_type;
        ParamMap params;

        friend class boost::serialization::access;
        template <class Archive>
            void serialize(Archive &ar, const unsigned int version) {
                ar & boost::serialization::base_object<BaseModel>(*this);
                ar & priors;
                ar & dist_type;
                ar & params;
            }

        shared_ptr<Distribution> init_distribution(string type) {
            if (type == "gaussian") {
                return shared_ptr<Distribution>(new GaussianDistribution);
            } else {
                return shared_ptr<Distribution>(new UniformDistribution);
            }
        }

        public:
        NaiveBayesModel(vector<double> prs, vector<string> ols,
                vector<string> pls, ParamMap ps, string dt): BaseModel(ols, pls) {
            priors = prs;
            params = ps;
            dist_type = dt;
        }

        NaiveBayesModel() {}

        virtual ~NaiveBayesModel() {}

        vector<double> eval(FeatureSet context);
    };

    class NaiveBayesTrainer : public Trainer<NaiveBayesModel> {
        bool continuous; // Are the feature values continous data
        shared_ptr<Distribution> dist;

        shared_ptr<Distribution> init_distribution(string type) {
            if (type == "gaussian") {
                return shared_ptr<Distribution>(new GaussianDistribution);
            } else {
                return shared_ptr<Distribution>(new UniformDistribution);
            }
        }

        public:
        NaiveBayesTrainer() {
            continuous = false;
        }

        NaiveBayesModel train(DataIndexer &di, ptree config);

        void set_heldout_data(EventSpace events);
    };
}

#endif

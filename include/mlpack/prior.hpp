/*
 * =====================================================================================
 *
 *       Filename:  prior.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/30/2011 12:41:58
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#ifndef PRIOR_H
#define PRIOR_H

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <map>
#include <vector>
#include <iterator>
#include <cmath>
#include <mlpack/events.hpp>

using namespace std;

namespace mlpack {
    class Prior {
        private:
            friend class boost::serialization::access;
            template <class Archive>
                void serialize(Archive &ar, const unsigned int version) {}

        public:
            virtual void log_prior(vector<double> &dist, FeatureSet &context) = 0;

            virtual void set_labels(vector<string> outcome_labels, vector<string> pred_labels) = 0;
    };

    class UniformPrior : public Prior {
        private:
            friend class boost::serialization::access;
            template <class Archive>
                void serialize(Archive &ar, const unsigned int version) {
                    ar & boost::serialization::base_object<Prior>(*this);
                    ar & n_outcomes;
                    ar & r;
                }

        public:
            int n_outcomes;
            double r;

            UniformPrior() {
            }

            void log_prior(vector<double> &dist, FeatureSet &context);

            void set_labels(vector<string> outcome_labels, vector<string> pred_labels);
    };
}

#endif

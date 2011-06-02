/*
 * =====================================================================================
 *
 *       Filename:  probs.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  02/06/11 15:25:17
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#ifndef MLPACK_PROBS_H
#define MLPACK_PROBS_H

#include <boost/property_tree/ptree.hpp>
#include <vector>
#include <map>
#include <cmath>
#include <mlpack/index.hpp>
#include <mlpack/events.hpp>
#include <mlpack/feature.hpp>

using boost::property_tree::ptree;
using namespace std;

namespace mlpack {
    typedef pair<int, int> Predicate; // A predicate is a pair of feature and outcome
    typedef map<Predicate, vector<double> > ParamMap;

    class Distribution {
        public:
            virtual ParamMap get_params(DataIndexer &di, ptree config) = 0;

            virtual double prob(ParamMap pmap, Feature &feat, int oid) = 0;
    };

    class GaussianDistribution : public Distribution {
        public:
            ParamMap get_params(DataIndexer &di, ptree config);

            double prob(ParamMap pmap, Feature &feat, int oid);
    };

    class UniformDistribution : public Distribution {
        public:
            ParamMap get_params(DataIndexer &di, ptree config);

            double prob(ParamMap pmap, Feature &feat, int oid);
    };
}

#endif

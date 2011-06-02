/*
 * =====================================================================================
 *
 *       Filename:  prior_test.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/31/2011 20:52:35
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#define BOOST_TEST_DYN_LINK
#ifdef STAND_ALONE
#define BOOST_TEST_MODULE Main
#endif

#include <boost/test/unit_test.hpp>
#include <fstream>
#include <mlpack/prior.hpp>
#include <cmath>

using namespace std;
using namespace mlpack;

BOOST_AUTO_TEST_SUITE(prior_test_suite)

BOOST_AUTO_TEST_CASE(prior_test) {
    UniformPrior prior;
    ifstream os("test/prior_test_outcomes");
    ifstream ps("test/prior_test_preds");
    vector<string> outcomes, preds;
    string s;
    while (os >> s) {
        outcomes.push_back(s);
    }
    while (ps >> s) {
        preds.push_back(s);
    }

    prior.set_labels(outcomes, preds);
    vector<double> dist;
    FeatureSet context;
    prior.log_prior(dist, context);

    vector<double>::iterator it;
    for (it = dist.begin(); it != dist.end(); it++) {
        double r = (*it);
        BOOST_CHECK_EQUAL(log(1.0 / 4.0), r);
    }
}

BOOST_AUTO_TEST_SUITE_END()

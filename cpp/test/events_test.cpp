/*
 * =====================================================================================
 *
 *       Filename:  events_test.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  31/05/11 16:21:05
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

#include <mlpack/events.hpp>
#include <mlpack/feature.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <boost/test/unit_test.hpp>

using namespace std;
using namespace mlpack;

BOOST_AUTO_TEST_SUITE(events_test_suite)

BOOST_AUTO_TEST_CASE(event_stream_test) {
    ifstream fs("test/test_events1");
    string line;
    FileEventStream stream("test/test_events1");
    int n_events = 0;
    while (stream.has_next()) {
        getline(fs, line);
        istringstream ss(line);
        string expected;
        ss >> expected;

        Event ev = stream.next();
        BOOST_CHECK(expected == ev.outcome);

        FeatureSet context = ev.context;
        FeatureIterator fit;
        int n_feats = 0;
        while (ss >> expected) {
            fit = context.find(expected);
            BOOST_CHECK(fit != context.end());
            n_feats++;
        }

        BOOST_CHECK_EQUAL(n_feats, context.feat_map.size());

        n_events++;
    }

    BOOST_CHECK_EQUAL(14, n_events);
}

BOOST_AUTO_TEST_SUITE_END()

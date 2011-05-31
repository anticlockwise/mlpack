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

#include <feature.hpp>
#include <events.hpp>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(events_test_suite)

BOOST_AUTO_TEST_CASE(event_stream_test) {
    FileEventStream stream("test/test_events1");
    while (stream.has_next()) {
        Event ev = stream.next();
        cout << ev.outcome << endl;
        FeatureSet context = ev.context;
        FeatureIterator fit;
        for (fit = context.begin(); fit != context.end(); fit++) {
            Feature f = (*fit).second;
            cout << f.name << ": " << f.value << " ";
        }
        cout << endl;
    }
}

BOOST_AUTO_TEST_SUITE_END()

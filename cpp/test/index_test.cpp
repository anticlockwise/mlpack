/*
 * =====================================================================================
 *
 *       Filename:  index_test.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/31/2011 21:42:12
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
#include <mlpack/events.hpp>
#include <mlpack/index.hpp>

using namespace std;

BOOST_AUTO_TEST_SUITE(index_test_suite)

BOOST_AUTO_TEST_CASE(index_test) {
    FileEventStream fes("test/test_events1");
    OnePassDataIndexer indexer(fes, 1, true);
    EventSpace contexts = indexer.contexts();
    BOOST_CHECK_EQUAL(14, contexts.size());

    FileEventStream fes2("test/test_events1");
    OnePassDataIndexer indexer2(fes2, 15, true);
    contexts = indexer2.contexts();
    BOOST_CHECK_EQUAL(0, contexts.size());

    FileEventStream fes3("test/test_events2");
    OnePassDataIndexer indexer3(fes3, 1, true);
    contexts = indexer3.contexts();
    BOOST_CHECK_EQUAL(13, contexts.size());
}

BOOST_AUTO_TEST_SUITE_END()

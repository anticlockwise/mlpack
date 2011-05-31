/*
 * =====================================================================================
 *
 *       Filename:  gistrainer_test.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/31/2011 23:07:23
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
#include <events.hpp>
#include <index.hpp>
#include <gistrainer.hpp>

using namespace std;

BOOST_AUTO_TEST_SUITE(gistrainer_test_suite)

BOOST_AUTO_TEST_CASE(gistrainer_test) {
    FileEventStream fes("test/test_events1");
    OnePassDataIndexer indexer(fes, 1, true);
}

BOOST_AUTO_TEST_SUITE_END()

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
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <fstream>
#include <mlpack/events.hpp>
#include <mlpack/index.hpp>
#include <mlpack/gistrainer.hpp>

using namespace std;
using boost::property_tree::ptree;

BOOST_AUTO_TEST_SUITE(gistrainer_test_suite)

BOOST_AUTO_TEST_CASE(gistrainer_test) {
    RealValueFileEventStream fes("test/real-valued-training-data.txt");
    OnePassDataIndexer indexer(fes, 1, true);
    ptree pt;
    read_ini("test/test_conf.ini", pt);
    GISTrainer trainer;
    trainer.train(indexer, NULL, pt);
}

BOOST_AUTO_TEST_SUITE_END()

/*
 * =====================================================================================
 *
 *       Filename:  maxent.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  03/06/11 12:28:37
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include <mlpack/events.hpp>
#include <mlpack/index.hpp>
#include <mlpack/gistrainer.hpp>
#include <mlpack/model.hpp>
#include <mlpack/prior.hpp>
#include <mlpack/command.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

using namespace std;
using boost::property_tree::ptree;
namespace po = boost::program_options;

namespace mlpack {
    class MaxentCommand : public MLCommand {
        private:
            shared_ptr<GISTrainer> trainer;
            MaxentModel model;

        public:
            MaxentCommand() {
                cmd_name = "maxent";
            }

            void init_trainer();
            void save_model(string fname, DataIndexer &di, ptree config);
            Model &load_model(string fname);
    };
}

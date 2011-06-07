/*
 * =====================================================================================
 *
 *       Filename:  command.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  03/06/11 10:46:00
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include <boost/shared_ptr.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/program_options.hpp>
#include <mlpack/model.hpp>
#include <mlpack/trainer.hpp>
#include <mlpack/events.hpp>
#include <mlpack/index.hpp>
#include <mlpack/feature.hpp>
#include <fstream>
#include <vector>

using boost::shared_ptr;
using boost::property_tree::ptree;
using namespace std;
namespace po = boost::program_options;

namespace mlpack {
    class Command {
        public:
            virtual int execute(int argc, char** argv) = 0;
    };

    class MLCommand : public Command {
        protected:
            string cmd_name;

        public:
            int execute( int argc, char ** argv);

            virtual void init_trainer() = 0;
            virtual void save_model(string fname, DataIndexer &di, ptree config) = 0;
            virtual Model &load_model(string fname) = 0;
    };
}

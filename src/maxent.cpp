/*
 * =====================================================================================
 *
 *       Filename:  maxent.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/30/2011 09:59:55
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include <mlpack/maxent.hpp>

namespace mlpack {
    void MaxentCommand::init_trainer() {
        trainer = shared_ptr<GISTrainer>(new GISTrainer);
    }

    void MaxentCommand::save_model(string fname, DataIndexer &di, ptree config) {
        MaxentModel model = trainer->train(di, config);
        ofstream ofs(fname.c_str());
        boost::archive::text_oarchive oa(ofs);
        oa << model;
    }

    Model &MaxentCommand::load_model(string fname) {
        ifstream ifs(fname.c_str());
        boost::archive::text_iarchive ia(ifs);
        ia >> model;
        return model;
    }
}

using namespace mlpack;

int main(int argc, char** argv) {
    MaxentCommand cmd;
    return cmd.execute(argc, argv);
}

/*
 * =====================================================================================
 *
 *       Filename:  gistrainer.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/30/2011 20:08:43
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */
#include "gistrainer.hpp"

GISModel GISTrainer::train(EventSpace es, ptree pt) {
    cutoff = pt.get<int>("cutoff");
    int iterations = pt.get<int>("iterations");

    contexts = es;
    vector<Event>::iterator it;
    int corr_const = 1;
    for (it = contexts.begin(); it != contexts.end(); it++) {
        FeatureSet &fset = (*it).context;
        FeatureMap &fmap = fset.feat_map;
        FeatureIterator fit;
        double cl = 0.0;
        for (fit = fmap.begin(); fit != fmap.end(); fit++) {
            cl += ((*fit).second).value;
        }
        if (cl > corr_const) {
            corr_const = (int)ceil(cl);
        }
    }
}

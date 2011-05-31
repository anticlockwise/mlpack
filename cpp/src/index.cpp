/*
 * =====================================================================================
 *
 *       Filename:  index.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/30/2011 22:36:29
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include "index.hpp"

vector<Event> BaseDataIndexer::contexts() {
    return events;
}

vector<string> BaseDataIndexer::pred_labels() {
    return plabels;
}

vector<string> BaseDataIndexer::outcome_labels() {
    return olabels;
}

vector<int> BaseDataIndexer::pred_counts() {
    return pcounts;
}

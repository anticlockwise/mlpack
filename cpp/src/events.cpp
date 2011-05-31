/*
 * =====================================================================================
 *
 *       Filename:  events.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/30/2011 23:15:24
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include "events.hpp"

Event RealValueFileEventStream::next() {
    string outcome;
    FeatureSet features;
    Feature feat;
    Event event;

    boost::char_separator<char> sep(" ");
    tokenizer tok(line, sep);
    tokenizer::iterator it = tok.begin();
    outcome = (*it);
    it++;
    while (it != tok.end()) {
        string context = (*it);
        size_t ind = context.find("=");
        feat.name = context.substr(0, ind);
        if (ind != string::npos) {
            feat.value = strtod(context.substr(ind+1).c_str(), NULL);
        }
        it++;
        features.put(feat);
    }

    event.outcome = outcome;
    event.context = features;
    event.count = 1;
}

bool cmp_event(const Event &e1, const Event &e2) {
    bool equals = true;

    FeatureMap fmap1 = e1.context.feat_map;
    FeatureMap fmap2 = e2.context.feat_map;
    FeatureIterator fit;
    for (fit = fmap1.begin(); fit != fmap1.end(); fit++) {
        string name = (*fit).first;
        if (fmap2.find(name) == fmap2.end()) {
            equals = false;
            break;
        } else {
            Feature &f1 = fmap1[name];
            Feature &f2 = fmap2[name];
            if (f1.value != f2.value) {
                equals = false;
                break;
            }
        }
    }
    return equals;
}

/*
 * =====================================================================================
 *
 *       Filename:  index.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/30/2011 20:47:56
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */
#ifndef DATAINDEXER_H
#define DATAINDEXER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include "events.hpp"

using namespace std;

class DataIndexer {
    public:
        virtual EventSpace contexts() = 0;

        virtual vector<string> pred_labels() = 0;

        virtual vector<int> pred_counts() = 0;

        virtual vector<string> outcome_labels() = 0;

        virtual int num_events() = 0;
};

class BaseDataIndexer : public DataIndexer {
    protected:
        EventSpace events;
        vector<string> plabels;
        vector<string> olabels;
        vector<int> pcounts;
        int num_events;

        int sort_n_merge(EventSpace &elist, bool s) {
            int num_uniq_events = 1;
            num_events = elist.size();
            if (num_events <= 1)
                return num_uniq_events;

            if (s) {
                EventSpace uniq_events;

                sort(elist.begin(), elist.end(), cmp_event);
                EventSpace::iterator eit = elist.begin();
                eit++;

                Event eprev = elist[0];
                uniq_events.push_back(eprev);
                while (eit != elist.end()) {
                    Event ecur = (*eit);
                    if (cmp_event(eprev, ecur)) {
                        Event &e = uniq_events[num_uniq_events-1];
                        e.count += ecur.count;
                    } else {
                        uniq_events.push_back(ecur);
                        num_uniq_events++;
                    }
                    eprev = ecur;
                }

                events = uniq_events;
            } else {
                num_uniq_events = num_events;
                events = elist;
            }

            return num_uniq_events;
        }

        void update(FeatureSet context, map<string, int> &pred_index,
                map<string, int> &counter, int cutoff) {
            FeatureIterator it;
            FeatureMap feat_map = context.feat_map;
            for (it = feat_map.begin(); it != feat_map.end(); it++) {
                string feat_name = it->first;
                if (counter.find(feat_name) == counter.end()) {
                    counter[feat_name] = 1;
                } else {
                    counter[feat_name] += 1;
                }

                Feature &feat = it->second;
                if (pred_index.find(feat_name) == pred_index.end()
                        && counter[feat_name] >= cutoff) {
                    int ind = pred_index.size();
                    pred_index[feat_name] = ind;
                }
            }
        }

    public:
        EventSpace contexts();
        vector<string> pred_labels();
        vector<string> outcome_labels();
        vector<int> pred_counts();
};

class OnePassDataIndexer : public BaseDataIndexer {
    public:
        OnePassDataIndexer(EventStream &stream, int cutoff=0, bool _sort=true);

    private:
        EventSpace compute_event_counts(EventStream &stream,
                map<string, int> &pred_index, int cutoff);
};

#endif

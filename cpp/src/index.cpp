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

EventSpace BaseDataIndexer::contexts() {
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

OnePassDataIndexer::OnePassDataIndexer(EventStream &stream, int cutoff, bool _sort) {
    map<string, int> pred_index;

    cout << "Indexing events using cutoff " << cutoff << endl;
    cout << "Computing event counts and indexing..." << endl;
    events = compute_event_counts(stream, pred_index, cutoff);
    cout << "DONE: " << events.size() << " in total" << endl;
    cout << "Sorting and merging events..." << endl;
    sort_n_merge(events, _sort);
    cout << "DONE: reduced to " << events.size() << " in total" << endl;
    cout << "Indexing done." << endl;
}

EventSpace OnePassDataIndexer::compute_event_counts(EventStream &stream,
        map<string, int> &pred_index, int cutoff) {
    EventSpace events;
    map<string, int> counter;
    map<string, int> oindex;

    while (stream.has_next()) {
        Event ev = stream.next();
        update(ev.context, pred_index, counter, cutoff);

        if (oindex.find(ev.outcome) == oindex.end()) {
            int ind = oindex.size();
            oindex[ev.outcome] = ind;
        }

        ev.oid = oindex[ev.outcome];
        FeatureSet new_set;
        FeatureSet old_set = ev.context;
        FeatureIterator fit;
        for (fit = old_set.begin(); fit != old_set.end(); fit++) {
            string fname = fit->first;
            if (pred_index.find(fname) != pred_index.end()) {
                Feature f = fit->second;
                f.id = pred_index[fname];
                new_set.put(f);
            }
        }

        if (new_set.size() > 0) {
            ev.context = new_set;
            events.push_back(ev);
        }
    }

    pcounts.resize(pred_index.size());
    plabels.resize(pred_index.size());
    olabels.resize(oindex.size());
    map<string, int>::iterator it;
    for (it = pred_index.begin(); it != pred_index.end(); it++) {
        string pred = (*it).first;
        int ind = (*it).second;
        pcounts[ind] = counter[pred];
        plabels[ind] = pred;
    }

    for (it = oindex.begin(); it != oindex.end(); it++) {
        int ind = (*it).second;
        string out = (*it).first;
        olabels[ind] = out;
    }

    return events;
}

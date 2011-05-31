/*
 * =====================================================================================
 *
 *       Filename:  events.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/30/2011 12:35:17
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#ifndef EVENTS_H
#define EVENTS_H

#include <vector>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <boost/tokenizer.hpp>
#include "feature.hpp"

using namespace std;

typedef boost::tokenizer<boost::char_separator<char> > 
    tokenizer;

struct Event {
    FeatureSet context;

    string outcome;

    int count;

    int oid;

    Event(FeatureSet f, string o, int c) {
        context = f;
        outcome = o;
        count = c;
        oid = -1;
    }

    Event() {
        outcome = "";
        count = 1;
        oid = -1;
    }
};

typedef vector<Event> EventSpace;

class EventStream {
    public:
        EventStream() {}

        virtual Event next() = 0;

        virtual bool has_next() = 0;
};

class FileEventStream : public EventStream {
    protected:
        ifstream input;
        string line;

    public:
        FileEventStream() {
        }

        FileEventStream(string fname) {
            input.open(fname.c_str(), ifstream::in);
        }

        Event next() {
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
                feat.name = (*it);
                feat.value = 1.0;
                it++;
                features.put(feat);
            }

            event.outcome = outcome;
            event.context = features;
            event.count = 1;

            return event;
        }

        bool has_next() {
            bool ret = getline(input, line);
            return ret;
        }
};

class RealValueFileEventStream : public FileEventStream {
    public:
        Event next();
};

bool cmp_event(const Event &e1, const Event &e2);

#endif

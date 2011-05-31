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
#include "feature.hpp"

using namespace std;

struct Event {
    FeatureSet context;

    string outcome;

    int count;

    Event(FeatureSet f, string o, int c) {
        context = f;
        outcome = o;
        count = c;
    }

    Event() {
        outcome = "";
        count = 0;
    }
};

typedef vector<Event> EventSpace;

class EventStream {
    public:
        virtual Event next() = 0;

        virtual bool has_next() = 0;
};

class FileEventStream : public EventStream {
    public:
        ifstream input;

        FileEventStream(string fname) {
            input.open(fname.c_str(), ifstream::in);
        }

        Event next();
        bool has_next();
};

bool cmp_event(const Event &e1, const Event &e2);

#endif

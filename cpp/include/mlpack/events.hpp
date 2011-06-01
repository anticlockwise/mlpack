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
#include <mlpack/feature.hpp>

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

    class FileBasedEventStream : public EventStream {
        protected:
            ifstream input;
            string line;

        public:
            FileBasedEventStream() {}
            FileBasedEventStream(string fname) {
                input.open(fname.c_str(), ifstream::in);
            }
            virtual Event next() {}
            bool has_next() {
                bool ret = getline(input, line);
                return ret;
            }
    };

    class PredicateEventStream : public FileBasedEventStream {
        public:
            PredicateEventStream() {}
            PredicateEventStream(string fname): FileBasedEventStream(fname) {}

            Event next();
    };

    class RealPredicateEventStream : public PredicateEventStream {
        public:
            RealPredicateEventStream() {}
            RealPredicateEventStream(string fname): PredicateEventStream(fname) {}

            Event next();
    };

    class FileEventStream : public FileBasedEventStream {
        public:
            FileEventStream() {}
            FileEventStream(string fname): FileBasedEventStream(fname) {}

            Event next();
    };

    class RealValueFileEventStream : public FileEventStream {
        public:
            RealValueFileEventStream(string fname): FileEventStream(fname) {}
            Event next();
    };

    bool cmp_event(const Event &e1, const Event &e2);

#endif

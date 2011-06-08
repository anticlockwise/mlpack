/*
 * =====================================================================================
 *
 *       Filename:  feature.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/30/2011 09:59:01
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#ifndef FEATURE_H
#define FEATURE_H

#include <string>
#include <map>
#include <iterator>

using namespace std;

namespace mlpack {
    struct Feature;

    typedef map<string, Feature> FeatureMap;
    typedef map<string, Feature>::iterator FeatureIterator;

    struct Feature {
        string name;
        double value;
        int id;

        Feature() {
            name = "";
            value = 0.0;
            id = -1;
        }

        Feature (string n, double v) {
            name = n;
            value = v;
            id = -1;
        }
    };

    struct FeatureSet {
        FeatureMap feat_map;

        map<string, double> attrs;

        void put_attr(string n, double a) {
            attrs[n] = a;
        }

        double get_attr(string n) {
            return attrs[n];
        }

        void put(Feature feat) {
            feat_map[feat.name] = feat;
        }

        Feature get(string name) {
            return feat_map[name];
        }

        FeatureIterator begin() {
            return feat_map.begin();
        }

        FeatureIterator end() {
            return feat_map.end();
        }

        FeatureIterator find(string name) {
            return feat_map.find(name);
        }

        size_t size() {
            return feat_map.size();
        }

        double length_sq() {
            double l = 0.0;
            FeatureIterator fit;
            for (fit = feat_map.begin(); fit != feat_map.end(); fit++) {
                Feature &f = fit->second;
                l += f.value * f.value;
            }
            return l;
        }
    };
}

#endif

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

struct Feature;

typedef map<string, Feature> FeatureMap;
typedef map<string, Feature>::iterator FeatureIterator;

struct Feature {
    string name;
    double value;

    Feature() {
        name = "";
        value = 0.0;
    }

    Feature (string n, double v) {
        name = n;
        value = v;
    }
};

struct FeatureSet {
    FeatureMap feat_map;

    Feature get(string name) {
        return feat_map[name];
    }

    FeatureIterator find(string name) {
        return feat_map.find(name);
    }

    size_t size() {
        return feat_map.size();
    }
};

#endif

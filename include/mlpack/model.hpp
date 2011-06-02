/*
 * =====================================================================================
 *
 *       Filename:  model.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  31/05/11 14:21:53
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#ifndef MODEL_H
#define MODEL_H

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <vector>
#include <algorithm>
#include <mlpack/feature.hpp>
#include <mlpack/prior.hpp>
#include <mlpack/params.hpp>

using namespace std;

namespace mlpack {
    class Model {
        private:
            friend class boost::serialization::access;
            template <class Archive>
                void serialize(Archive &ar, const unsigned int version) {}

        public:
            virtual vector<double> eval(FeatureSet context) = 0;

            virtual string best_outcome(vector<double> outcomes) = 0;

            virtual string outcome(int i) = 0;

            virtual int index(string out) = 0;

            virtual int pred_index(string pred) = 0;
    };

    class BaseModel : public Model {
        friend class boost::serialization::access;
        template <class Archive>
            void serialize(Archive &ar, const unsigned int version) {
                ar & boost::serialization::base_object<Model>(*this);
                ar & olabels;
                ar & pmap;
            }
        protected:
            vector<string> olabels;
            map<string, int> pmap;

        public:
            BaseModel() {}

            BaseModel(vector<string> ols, vector<string> pls) {
                olabels = ols;
                int n = pls.size(), i;
                for (i = 0; i < n; i++) {
                    pmap[pls[i]] = i;
                }
            }

            string best_outcome(vector<double> outcomes);
            string outcome(int i);
            int index (string out);
            int pred_index(string pred);
    };

    class MaxentModel : public BaseModel {
        private:
            friend class boost::serialization::access;
            template <class Archive>
                void serialize(Archive &ar, const unsigned int version) {
                    ar & boost::serialization::base_object<BaseModel>(*this);
                    ar & maxent_params;
                    ar & prior;
                }

        protected:
            MaxentParameters *maxent_params;
            Prior *prior;

        public:
            MaxentModel() {
                prior = NULL;
                maxent_params = NULL;
            }

            MaxentModel(vector<Parameters> params, vector<string> pls,
                    vector<string> ols, Prior *p): BaseModel(ols, pls) {
                prior = p;
                maxent_params = new MaxentParameters(params, 0.0, 1.0, ols.size());
            }

            static vector<double> eval(FeatureSet context, vector<double> &prior,
                    MaxentParameters model) {
                vector<Parameters> params = model.params;
                vector<int> n_feats(model.n_outcomes);
                vector<int> active_outcomes;
                vector<double> active_params;
                double value = 1.0;
                int ai, n_outcomes, oid;
                FeatureIterator fit;
                for (fit = context.begin(); fit != context.end(); fit++) {
                    Feature f = (*fit).second;
                    if (f.id != -1) {
                        Parameters &p = params[f.id];
                        active_outcomes = p.outcomes;
                        active_params = p.params;
                        value = f.value;
                        n_outcomes = active_outcomes.size();
                        for (ai = 0; ai < n_outcomes; ai++) {
                            oid = active_outcomes[ai];
                            n_feats[oid]++;
                            prior[oid] += active_params[ai] * value;
                        }
                    }
                }

                double normal = 0.0;
                for (oid = 0; oid < model.n_outcomes; oid++) {
                    if (model.corr_param != 0) {
                        prior[oid] = exp(prior[oid] * model.const_inverse
                                + ((1.0 - ((double)n_feats[oid] / model.corr_constant))
                                    * model.corr_param));
                    } else {
                        prior[oid] = exp(prior[oid] * model.const_inverse);
                    }
                    normal += prior[oid];
                }

                for (oid = 0; oid < model.n_outcomes; oid++) {
                    prior[oid] /= normal;
                }

                return prior;
            }

            vector<double> eval(FeatureSet context);
    };
}

#endif

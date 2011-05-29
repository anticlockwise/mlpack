package com.mlpack.maxent;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import com.mlpack.Counter;
import com.mlpack.Feature;
import com.mlpack.FeatureSet;

public class MaxentModel {
    public static Map<String, Counter> eval(FeatureSet context,
            Map<String, Counter> priors, Map<String, Parameters> params) {
        Set<String> outcomes = null;
        Map<String, Counter> numFeats = new HashMap<String, Counter>();
        double value = 1.0;

        for (Feature feat : context) {
            Parameters param = params.get(feat.getName());
            outcomes = param.getOutcomes();
            value = feat.getValue();
            for (String outcome : outcomes) {
                Counter count = null;
                if (!numFeats.containsKey(outcome)) {
                    count = new Counter();
                    numFeats.put(outcome, count);
                } else {
                    count = numFeats.get(outcome);
                }
                count.addCount(1);

                count = null;
                if (!priors.containsKey(outcome)) {
                    count = new Counter();
                    priors.put(outcome, count);
                } else {
                    count = priors.get(outcome);
                }
                count.addCount(param.getParam(outcome) * value);
            }
        }

        double normal = 0.0;
        outcomes = priors.keySet();
        for (String outcome : outcomes) {
            normal += priors.get(outcome).getCount();
        }

        for (String outcome: outcomes) {
            Counter count = priors.get(outcome);
            count.setCount(count.getCount() / normal);
        }

        return priors;
    }
}

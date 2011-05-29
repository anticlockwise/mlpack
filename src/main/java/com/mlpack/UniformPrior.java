package com.mlpack;

import java.util.Map;
import java.util.Set;

public class UniformPrior implements Prior {
    private Set<String> outcomes;

    private double      r;

    public UniformPrior(Set<String> outcomes) {
        this.outcomes = outcomes;
        int numOutcomes = outcomes.size();
        r = Math.log(1.0 / numOutcomes);
    }

    public void logPrior(Map<String, Counter> dist, FeatureSet context) {
        for (String outcome : outcomes) {
            dist.put(outcome, new Counter(r));
        }
    }
}

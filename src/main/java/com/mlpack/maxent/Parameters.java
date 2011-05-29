package com.mlpack.maxent;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class Parameters {
    Map<String, Double> params = new HashMap<String, Double>();

    public void setParameter(String param, double value) {
        params.put(param, value);
    }

    public double getParam(String param) {
        return params.get(param);
    }

    public Set<String> getOutcomes() {
        return params.keySet();
    }

    public boolean contains(String param) {
        return params.containsKey(param);
    }
}

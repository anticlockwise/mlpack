package com.mlpack;

import java.util.Map;

public interface Prior {
    public void logPrior(Map<String, Counter> dist, FeatureSet context);
}
